# OATS + 双分支 AWQ

本实现只从 `awq_methods.zip` 移植量化逻辑。剪枝和低秩分解仍调用本仓库的 OATS：激活加权的 `torch.linalg.svd(..., full_matrices=False)`、原来的奇异值分配方式、先低秩后稀疏更新、原来的 2:4 投影、零稀疏初始化。默认执行 80 轮并返回最后一轮，没有提前停止或最优轮选择。

支持 Llama2-7B 风格的独立 Q/K/V、无 GQA 架构，rank 32/128，2:4 稀疏主干。`prepare` 是原始 OATS 基线；`posthoc` 对已完成的 OATS 模型量化；`inloop` 将量化插入原 OATS 更新步骤。Embedding、LM head 和 bias 不做低比特量化。Post-hoc 的等价尺度融合会调整相邻层、LayerNorm 和相关 bias。

## 算法边界

模型使用原生 `CompressedLinear`：

```text
y = linear(x, S, bias) + linear(linear(x, V), U)
V = 附件 first / A / lr.l1.weight
U = 附件 second / B / lr.l2.weight
```

In-loop 在原 OATS `altern_ls` 中插入两次量化，`D = sqrt(clamp(diag_approx, 1e-8))`：

```text
S_q = 0
重复 num_iters 次：
    在 (W - S_q) D 上执行原 OATS 完整 SVD，截取 rank
    按原 OATS 方式得到 U，以及还原到原坐标的 V
    V_q, U_q = AWQ 搜索并量化 V、U；目标为 (W - S_q)X
    对 (W - U_q V_q)D 执行原 OATS 2:4 投影
    将稀疏结果还原到原坐标 S
    S_q = AWQ 搜索并量化 S；目标为 (W - U_q V_q)X
返回最后一轮的 V_q、U_q、S_q
```

SVD 始终使用 FP32。量化因子按模型存储 dtype 搜索、舍入，再以 FP32 参与 OATS 残差更新。所有 AWQ 量化都在原坐标中进行；诊断误差不影响更新或返回轮次。激活采样不改变原 OATS 使用完整校准集积累的对角统计。

Post-hoc 顺序与附件一致：等价预平衡 → 外部尺度搜索 → rank 通道尺度搜索 → clipping → 最终量化。保留 Q/K 的 clipping 跳过规则，以及 `U` 使用量化后 `V` 输出做 clipping 的行为。每个块向下一块传递其量化前输出，不在量化后重新分解。

| 配置 | Post-hoc | In-loop |
|---|---|---|
| S / V | W4G128 | W4G128 |
| U | W4G32 | rank 32: W4G32；rank 128: W4G128 |
| Scale grid | 20 | 20 |
| Scale 刷新 | 最终量化阶段 | V/U 每轮，S 第 1/11/21…轮 |
| Clipping | 开启；grid 20，max shrink 0.5 | 关闭 |
| 量化校准 | PileVal：seed 42，128 段合格文本拼接为 512-token 块 | OATS C4 输入，每个投影最多采 512 tokens |
| 选轮 | 不适用 | 最后一轮 |

两种量化模式的校准、分组和 clipping 不同，不能把结果解释为仅改变量化时机的消融。

保存的是假量化后的浮点权重，不是打包 INT4；不提供 INT4 内核、实际压缩率或推理加速承诺。稀疏量化保留现有零掩码，对活跃元素被舍入成零的情况采用附件的非零投影。分组不可整除、非有限数值、dtype 转换破坏掩码等情况明确报错。

## 最多四张 GPU 的投影并行

`prepare` 和 `inloop` 支持 `--devices`，最多四张不同的 GPU。每个 Transformer block 先在校准卡上收集完整统计，再把七个投影分配给常驻进程；较大的矩阵优先调度。每个任务调用同一个原生 `altern_ls`，不会切分矩阵做独立 SVD。全部投影成功后，主进程按原顺序安装结果、写日志并重新前向，再处理下一个 block。

仅使用本机物理 GPU **4、5、6、7**：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv-awq/bin/python run_oats_awq.py inloop \
  --model /path/to/Llama-2-7b-hf --rank 32 \
  --calibration-path /home/zwang53/oats_code/calib_llama2-7b.pth \
  --output-dir runs_awq/r32/inloop_four_gpu \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 --num-iters 80 --dtype fp16
```

设置可见设备后，进程内 `cuda:0/1/2/3` 分别对应物理 `4/5/6/7`。默认第一张选定卡也承担校准前向；可通过 `--device` 从这四张卡中另选校准卡，不会额外使用第五张卡。`--worker-cpu-threads` 默认 1，避免多个进程争抢 CPU 线程。

同样的参数可用于 `prepare`。Post-hoc 和 PPL 评估仍为单卡，不接受 `--devices`。不传 `--devices` 时保留单设备执行路径。并行配置及 `CUDA_VISIBLE_DEVICES` 记录在 manifest 中，续跑需保持一致。若任一 worker 失败，停止其余 worker；当前 block 不提交，之后可从上一个完成的 block 恢复。

进程启动、CPU/GPU 传输和校准前向均有开销，小模型可能更慢。实际 7B 加速比需另行测量；回归测试只验证方法与结果一致，不用小模型计时推断完整实验的加速倍数。

## 环境与测试

可在已有 PyTorch 环境中安装最小依赖，不需要 `lm_eval` 或 AWQ CUDA 扩展：

```bash
python -m pip install -r requirements-awq.txt
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m unittest discover -s tests -p 'test_oats_awq*.py' -v
```

本次验证环境为 Python 3.13.2、PyTorch 2.13.0+cu130、Transformers 5.15.0。使用隔离的 `.venv-awq`（继承宿主 PyTorch），可直接使用 `.venv-awq/bin/python` 运行小模型测试。测试覆盖 CPU FP32 与 CUDA FP16、两层随机 Llama、两个 rank、七个投影、原算法回归、量化反馈、完整重载及断点恢复。没有下载或运行 7B 模型。

## 全模型实验命令

以下命令由用户实际启动实验；每个运行使用独立目录。基础模型可以是本地 HF 目录或 `meta-llama/Llama-2-7b-hf`。需要该模型的访问权限和权重，以及校准/评测数据。

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
OATS_BASE=/path/to/Llama-2-7b-hf
OATS_CAL=/home/zwang53/oats_code/calib_llama2-7b.pth

for rank in 32 128; do
  python run_oats_awq.py prepare --model "$OATS_BASE" --rank "$rank" \
    --output-dir "runs_awq/r${rank}/prepare" --dtype fp16 --device cuda:0 \
    --calibration-path "$OATS_CAL" --num-iters 80

  python run_oats_awq.py posthoc --model "$OATS_BASE" --rank "$rank" \
    --checkpoint "runs_awq/r${rank}/prepare/checkpoint.pt" \
    --output-dir "runs_awq/r${rank}/posthoc" --dtype fp16 --device cuda:0

  python run_oats_awq.py inloop --model "$OATS_BASE" --rank "$rank" \
    --output-dir "runs_awq/r${rank}/inloop" --dtype fp16 --device cuda:0 \
    --calibration-path "$OATS_CAL" --num-iters 80

  for method in prepare posthoc inloop; do
    python run_oats_awq.py eval --model "$OATS_BASE" --rank "$rank" \
      --checkpoint "runs_awq/r${rank}/${method}/checkpoint.pt" \
      --output-dir "runs_awq/r${rank}/${method}_eval" --device cuda:0 \
      --ppl-seqlen 128
  done
done
```

`prepare/inloop` 不指定 `--calibration-path` 时使用本仓库现有 C4 loader，默认 128×2048、seed 0、batch 32。指定文件时使用其中的全部样本，保持原有 tuple/dict 解析方式，不因 `--cal-nsamples` 而截断；实际样本数会被记录。两种优化流程应使用相同的校准文件、dtype、rank 和迭代数。

Post-hoc 使用附件的 PileVal 采样规则，128 指文本数量，不保证生成 128 个 token 块。其 `--calibration-path` 是用于离线验证的二维整数 `input_ids` tensor，与优化模式的文件格式不同。

`eval` 默认评测 WikiText-2 raw test，非重叠序列，batch 1；按实际 shifted target 数累计 NLL。可用 `--eval-tokens` 指定本地 token tensor进行离线测试。评测保持检查点保存的 dtype，不再次量化或施加尺度。

现有 OATS 检查点也可用于 post-hoc：`--checkpoint /path/to/oats_layer_directory`。目录必须包含完整的 `prune_chkpt.pt` 和所有 `oats_chkpt_*.pt`；`--rank`、`--dtype` 必须与实际张量一致，不能仅依据目录名字推断。

## 产物与恢复

- `checkpoint.pt`：模型配置、完整 state_dict、实际 RoPE 非持久缓存、rank、dtype、方法和量化元数据。重载后仍使用原生 `CompressedLinear`；尺度已应用，不能再应用一次。
- `quant_report.json`：配置、掩码审计及量化信息；`iterations.jsonl` 保存优化逐轮诊断，明确标记 `iterate_selection=last`。
- `manifest.json`：运行设置、输入文件标识、源代码来源。使用相同命令加 `--resume` 恢复中断任务；设置不一致或运行已经完成时拒绝恢复。
- `layers/`：原 OATS 格式的分层检查点。Post-hoc 另存已完成的量化层，恢复时从未量化源模型重放校准前向，然后装入已完成层。
- `metrics.json`：评估结果，仅由 `eval` 产生。

Post-hoc 必须保留完整模型状态。旧的仅含 `sp.weight/lr.l1.weight/lr.l2.weight` 的参数字典缺少被融合修改的相邻层和 LayerNorm，不能用于完整重建这个量化产物。

来源、文件校验和及固定 AWQ revision 位于 `pruning_methods/OATS/awq/SOURCES.json`；许可证位于其 `reference/` 子目录。附件的 `fit_dual_awq`、Wanda 剪枝器和 SVD 优化器未被移植。

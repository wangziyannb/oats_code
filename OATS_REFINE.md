# OATS 固定掩码的整数码联合优化

独立研究入口 `run_oats_refine.py`。原 OATS 产生初始 S/U/V 和 2:4 掩码，新量化器只调整三个矩阵的整数码及正的分组步长。不执行 SVD，不选择新掩码，不改变 rank，不修改已有 `prepare/posthoc/inloop` 路径。

`run_oats_refine.py` 提供首个 Transformer block 的投影对照；`run_oats_refine_full.py` 提供逐块全模型量化、断点续跑和完整重载后的 PPL 评估。两者使用相同的整数码优化器。

## 目标及更新

固定 `h = diag(H) > 0`，优化 `F = 0.5 * sum((W-S-U@V)^2 * h)`。工作精度为 FP32（数值验证可用 FP64），诊断额外测量 FP16 推理输出。

- **单坐标更新**：按当前双分支残差精确更新合法整数码。U 的曲率来自 `V H V.T`，V 的曲率来自 `U.T U` 和 h。按 latent channel 顺序更新；U 的输出行、V 的输入列可以并行，不能用同一份旧梯度同时更新所有 latent channels。
- **成对更新**：固定 V，对 S/U 的两个整数码联合搜索；随后固定 U，对 S/V 联合搜索。枚举 S 的 16 个码（排除零码），对另一个变量做条件最优舍入。按耦合强度筛选候选位置，每个 latent channel、每个独立行/列实际更新至多一对。`--pair-candidates` 控制寻找一个合法位置的候选池大小，不是精确评价多个位置后选优。
- **尺度更新**：固定码和零点，对 group step 解一维加权最小二乘。稀疏步长下界还确保任一活跃位置的一步非零码在 FP16 存储时不会下溢为零。
- **数值保护**：每个完整更新阶段都重新反量化并计算 F，只有 F 不增加、FP16 有限性和稀疏掩码检查通过时才提交。拒绝的阶段整体回退。日志保留候选目标和回退原因字段。

本版不支持非对角 H、不更新 zero point、不更新掩码、不搜索新的通道变换。H 和通道变换在各方案间相同。收敛论断仅针对固定投影的重构目标值，不是全局最优或 PPL 保证。

## 五组对照

所有组使用相同的原 OATS 初始化及 AWQ 初始化。该 AWQ 初始化复用量化尺度搜索：S/V W4G128，U W4Gmin(rank,128)，20 个尺度候选，无 clipping、无相邻层融合。它与之前完整模型的 Post-hoc AWQ 配置不同，不能混用名称或指标。

1. `awq`：AWQ 风格的残差尺度搜索，形成显式整数码和固定通道变换。
2. `coordinate`：从 AWQ 出发，3 轮单坐标码更新和尺度更新。
3. `coordinate_then_codes`：从第 2 组终态出发，额外做一次 U/V/S 普通整数码更新；不再更新尺度，作为额外计算的对照。
4. `coordinate_then_pair`：从第 2 组终态出发，额外做一次 S/U 和一次 S/V 成对更新；不再更新尺度，用于检查成对步骤的独立收益。
5. `paired`：从同一 AWQ 初始化出发，3 轮单坐标、成对、尺度更新。

第 2/5 组匹配轮数但不匹配计算预算；第 3/4 组均使用第 2 组再加额外计算，但两类更新的耗时仍不同。逐阶段时间和目标均保存，不能将结果称为等计算量比较。

## 数据与运行

从同一基础 Llama2 模型的第一个 block 收集七个投影的输入。默认使用原 128 段 C4 校准文件：前 64 段用于新量化器，后 64 段用于留出诊断；每部分均匀采 512 个激活 token。对角统计使用前 64 段的全部 token。

后 64 段仅对**新量化器**留出：原 OATS 已经使用过全部 128 段，因此这不是对整个压缩流程完全独立的测试集。投影指标来自相同的 dense first-block 输入，不是把七个压缩投影串起来的 block 输出指标。

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

.venv-awq/bin/python run_oats_refine.py capture \
  --model /path/to/Llama-2-7b-hf \
  --calibration-path calib_llama2-7b.pth \
  --output-dir runs_refine/example/capture --device cuda:0

.venv-awq/bin/python run_oats_refine.py benchmark \
  --capture-dir runs_refine/example/capture \
  --baseline-dir runs_awq/llama2_r32_then_r128_20261006_185158 \
  --output-dir runs_refine/example/comparison \
  --ranks 32 128 --devices cuda:0 cuda:1 cuda:2 cuda:3 --sweeps 3

CUDA_VISIBLE_DEVICES=4,5,6,7 .venv-awq/bin/python -m unittest discover \
  -s tests -p 'test_oats_refine.py' -v
```

`--baseline-dir` 下应包含 `r32/prepare` 和 `r128/prepare` 的原生分层及完整检查点。校准文件、基础模型、dtype 和 rank 会检查一致性。GPU 数量上限为四张；各 rank 按参数顺序运行，rank 内投影并行。输出必须使用空目录，本版不自动续跑。

## 保存与重载

各投影目录保存五个 `.pt` 产物、`metrics.json`、`phases.jsonl`；总目录保存 `manifest.json`、`status.json`、`results.json` 和本次源码快照。显式量化产物包含 uint8 保存的 W4 码、组步长、零点、固定 row/column gain 和原稀疏掩码，未打包 INT4。

```python
import torch
from pruning_methods.OATS.awq.refine import CodeMatrix

payload = torch.load('coordinate.pt', map_location='cpu', weights_only=True)
grids = {name: CodeMatrix.from_state_dict(state)
         for name, state in payload['grids'].items()}
S, U, V = (grids[name].values() for name in ('S', 'U', 'V'))
# values() 已还原原权重坐标，不能再次施加 row/column gain。
```

## 首层实测结果（2026-10-07）

最终实验目录：[comparison_final](runs_refine/first_block_20261007/comparison_final/)。其中 [summary.json](runs_refine/first_block_20261007/comparison_final/summary.json) 包含汇总，[projection_metrics.csv](runs_refine/first_block_20261007/comparison_final/projection_metrics.csv) 包含各投影的绝对误差与时间。`comparison_v1` 和 `comparison_v2_fp16` 是探索记录，最终结论使用 `comparison_final`。

使用物理 GPU 4–7，先完成 rank 32，再运行 rank 128；每个 rank 的七个投影并行。下表为留出输入上的 **FP16 前向输出 MSE**，降幅按七个投影的误差比取几何平均：`100 * (1 - geometric_mean(after / before))`。每个投影等权，不代表整个 block 的输出误差或全模型 PPL。

| rank | AWQ → 普通坐标更新 | AWQ → 交替成对更新 | 成对更新相对普通坐标更新的额外降幅 | 额外改善的投影数 |
|---|---:|---:|---:|---:|
| 32 | 64.46% | 64.92% | 1.32% | 7/7 |
| 128 | 56.88% | 60.16% | 7.60% | 6/7 |

末尾追加计算的对照并不一致：在普通坐标优化后，只追加一轮普通码更新，rank 32/128 的留出误差分别增加 0.72%/0.67%；只追加一次成对更新，分别增加 0.27%/降低 0.23%。这些更新都降低了训练目标。因此，现阶段有证据支持继续测试交替成对更新，但不能声称成对步骤必然改善留出误差，也不能把匹配轮数的结果当作匹配计算量的收益。

验证完成：新增 9 项 CPU/CUDA 测试和原 OATS 的 9 项核心回归测试全部通过；14 个投影的 FP16 稀疏掩码均保持完整；56 个 refinement 产物重载后的反量化权重逐位一致；658 个阶段记录中没有已提交的目标上升，本次没有触发回退。测试日志保存在最终实验目录的 `validation/`。

上述首层探针未计算完整模型 PPL；随后完成的全模型实验见下文。等时间预算的对照尚未完成；当前留出数据的适用边界见上文。

## 已有工作与方法定位

整数坐标优化及单调重构下降本身不是新贡献。相关工作包括 [CALDERA](https://arxiv.org/abs/2405.18886)、[SLiM](https://arxiv.org/abs/2410.09615)、[GPTQ-intrinsic LoRA / Bid-Up](https://arxiv.org/abs/2606.01412)。本原型检验的是固定 2:4 掩码、两个因子均低比特时，跨分支成对整数码更新是否在单坐标更新之外带来稳定收益；尚未确立研究新颖性或全模型精度收益。

## 全模型实验

全模型入口复用已完成的原生 OATS `prepare` 分层检查点。三组方案为 `awq`（只有本原型的 AWQ 初始化）、`coordinate`（三轮单码及步长优化）、`paired`（三轮单码、成对及步长优化）。每组均固定原 OATS 的掩码和 rank，不重新运行 SVD。它是完成 OATS 后的量化变量优化，不是修补原来的 80 轮 SVD→AWQ in-loop 实现；与旧 in-loop 的比较用于检验替代优化方案的完整模型效果。

每个 block 使用已量化前缀产生的输入，在当前 dense block 上收集七个投影的统计；用全部 128 段原 C4 校准序列的 token 计算 H，按序列均匀抽取 512 token 做 AWQ 搜索。随后加载对应的 OATS 因子，量化并重新前向，将实际量化输出传给下一个 block。各组使用相同初始化算法；由于前缀不同，后续层的校准轨迹及初始化数值可以不同。此处使用全部 128 段，与首层探针的 64/64 切分不同。

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

.venv-awq/bin/python run_oats_refine_full.py quantize \
  --model /path/to/Llama-2-7b-hf \
  --baseline-dir runs_awq/llama2_r32_then_r128_20261006_185158/r32/prepare \
  --calibration-path calib_llama2-7b.pth \
  --output-dir runs_refine/example_full/r32/paired \
  --rank 32 --arm paired --dtype fp16 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 --batch-size 8 \
  --tokens 512 --scale-grid 20 --sweeps 3 --pair-candidates 16

.venv-awq/bin/python run_oats_refine_full.py eval \
  --checkpoint runs_refine/example_full/r32/paired/checkpoint.pt \
  --eval-tokens /path/to/wikitext2_raw_test_tokens.pt \
  --output-dir runs_refine/example_full/r32/paired_eval \
  --device cuda:0 --ppl-seqlen 128
```

更换 `--rank`、基线目录和输出目录即可运行 rank 128。`--arm coordinate` 或 `--arm awq` 运行其余对照。`run_oats_refine_study.py` 提供 rank 32→128 的自动队列；每个 rank 先依次量化三组，再在独立 GPU 上并行运行 batch 1 PPL。

队列支持 `--resume --code-dir /path/to/frozen/code`：校验命令后跳过已完成产物，并为后续任务记录当前物理 GPU 编号。只能在没有运行中的旧队列时接续；单个尚未完成的量化任务仍要求原 manifest 完全匹配，不能在中途静默改变设备配置。

每个 block 保存完整状态和七个量化码产物；最后保存完整模型、模型配置、rank、dtype、量化信息及非持久 rotary buffers。评估先验证重载后的 probe logits 逐位相同，再调用与旧实验相同的 PPL 函数。输出目录与旧实验隔离。中断后可用原参数加 `--resume` 续跑；manifest 检查方法、配置、来源文件身份和源码哈希，已提交的 block 直接重载并重新传播输入，不重复量化或施加尺度。

### 完整 Llama2-7B 结果（2026-10-07）

实验目录：[full_model_20261007_0848](runs_refine/full_model_20261007_0848/)。原始 PPL 位于 `results.json`；与旧实验的合并对照位于 [comparison.json](runs_refine/full_model_20261007_0848/comparison.json) 和 [comparison.csv](runs_refine/full_model_20261007_0848/comparison.csv)。两个 rank 均先完成所有量化及评估，再进入下一个 rank。

所有模型均使用 FP16、2:4；WikiText-2 raw test，长度 128、batch 1，预测 338,709 个 token。PPL 越低越好。

| 方法 | rank 32 | rank 128 |
|---|---:|---:|
| 原 OATS，未量化（旧结果） | 24.0831 | 17.0228 |
| 原 Post-hoc（旧结果） | 24.4540 | 18.2751 |
| 原 In-loop（旧结果） | 45.4444 | 30.7496 |
| 本轮 AWQ 初始化 | 29.6735 | 23.3335 |
| 普通坐标优化，3 轮 | 27.7661 | 20.2274 |
| 交替成对优化，3 轮 | 28.0701 | 19.8339 |

相对本轮 AWQ 初始化，普通坐标优化使 PPL 分别降低 6.43%/13.31%；成对方案分别降低 5.40%/15.00%。相对普通坐标，成对方案在 rank 32 上变差 1.09%，在 rank 128 上改善 1.95%。因此，整数码优化在本轮初始化上有收益，成对步骤尚无跨 rank 的稳定 PPL 优势；两种方案仍未达到旧 Post-hoc 的结果。

这不能直接归结为修复了原 In-loop：本轮从已经完成的浮点 OATS 开始，固定其掩码，而旧 In-loop 在每轮 SVD/稀疏更新中量化反馈。此外，旧 Post-hoc 与本轮的校准数据、预平衡、尺度融合、clipping、rank 128 的 U 分组等配置不同。旧结果仅作为完整模型参照，不是纯时机消融，也不是对成对步骤的单独归因。匹配轮数也不等于匹配计算预算。

21 项测试通过。六个模型共 1,344 个投影的掩码及码产物重载均有效；18,816 个优化阶段中没有已提交的目标上升，本次没有回退；六个完整模型的 probe logits 在重载后逐位一致，全部 PPL 有限。详见 [validation_summary.json](runs_refine/full_model_20261007_0848/validation_summary.json)。

GPU 记录：rank 32 的全部阶段及 rank 128 的 AWQ 初始化使用物理 0–3；随后按用户要求切换，rank 128 的普通坐标、成对优化及评估使用物理 4–7 范围内的设备。切换时让当前任务完成后再接续，未重跑已完成实验。原运行源码保存在 `code/`；队列接续源码及交接记录保存在 `control_v2/`。由于中途其他用户的 GPU 负载及设备分配变化，本次墙钟时间不用于方法速度比较。

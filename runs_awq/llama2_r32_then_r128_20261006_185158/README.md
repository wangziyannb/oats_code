# Llama2-7B：rank 32 → rank 128

本次实验由 `run_sequence.py` 串行调度，运行会话 `oats_awq_r32_r128_20261006`。

每个 rank 的顺序：prepare → posthoc → inloop → prepare_eval → posthoc_eval → inloop_eval。r=32 全部成功后才开始 r=128；任一阶段失败后停止，不跳过失败任务。

配置：FP16、2:4、80 轮、最后一轮返回；prepare/inloop 仅在物理 GPU 4/5/6/7 上做投影并行，posthoc/eval 使用物理 GPU 4。量化搜索 grid=20，inloop 采样 512 激活 tokens、稀疏尺度每 10 轮刷新。

基础模型和校准数据详情见 `data_preflight.json`。OATS 使用已有 128×2048 校准文件；posthoc 使用 PileVal seed 42 的 128 段合格文本（65×512 token 块）；WikiText-2 raw test 使用长度 128、batch 1。数据已预缓存，任务离线加载。

- `status.json`：当前阶段、PID、每阶段状态和错误，每 15 秒刷新。
- `queue.log`：阶段起止信息。
- `logs/`：各阶段完整日志。
- `commands.json`：完整命令与运行环境。
- `code/` 与 `source_sha256.json`：启动时冻结的源码与校验和，两个 rank 使用相同实现。
- `r32/`、`r128/`：各方法模型、校准续跑检查点和 PPL 指标。
- `results.json`：全部阶段成功后汇总六组评测结果。

查看当前模型处理日志：

```bash
tail -f logs/r32_prepare.log
```

队列拒绝重复启动；续跑需先检查失败日志和已完成产物，使用该阶段 `commands.json` 中的原命令配合原生 `--resume`，不能覆盖已完成目录。

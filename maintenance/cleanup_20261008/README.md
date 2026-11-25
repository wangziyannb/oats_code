# 模型归档与空间清理（2026-10-08）

本次只整理保存格式、删除重复权重与中间缓存，不重新训练、剪枝、分解、量化或计算完整 PPL。

## 保留内容

- `runs_awq/llama2_r32_then_r128_20261006_185158/r{32,128}/{prepare,posthoc,inloop}/checkpoint.pt`：6 份原始完整检查点，文件不修改。
- `runs_refine/full_model_20261007_0848/r{32,128}/{awq,coordinate,paired}/checkpoint.pt`：6 份原始完整检查点，文件不修改。
- 10 个历史 OATS 导出目录中的 `checkpoint.pt`：从原层检查点合并而成的完整模型，包含模型配置、所有参数、模块 rank 和非持久化 buffer。
- 源码、环境、配置、实验 JSON/CSV、日志、逐轮诊断，以及小体积的原始校准输入和重载探针。

历史模型包括 Llama2-7B 的 rank 32/64/128，Llama3.1-8B 的 rank 32/128，Qwen3-1.7B-Base 的 rank 32/128，以及 Qwen3-8B-Base 的 rank 32/64/128。最终文件列表和空间统计见 `summary.json`。

## 核对与删除范围

`recent_cleanup.json` 记录了 12 份近期模型与全部 384 份层检查点的逐张量精确比较。确认一致后，删除重复的 OATS 层文件、post-hoc 层文件、refinement 层文件、量化码状态、续跑激活缓存和首层试验张量。现有 JSON/CSV 实验结果与诊断日志保留。

`legacy_cleanup.json` 逐文件记录历史归档的来源和删除状态。每份历史模型经过以下检查后才删除原 `param_dict`、`oats_chkpt_*` 和 `prune_chkpt.pt`：

1. 检查层数、rank、矩阵形状、FP16、有限值、2:4 稀疏约束。
2. 所有 S/U/V 与旧导出逐张量精确一致；所有未压缩的层内 norm 与指定基础模型快照精确一致。
3. embedding、最终 norm、输出头从对应本地基础模型快照补齐；保留原有层内所有参数。
4. 严格重载全部参数和 buffer，逐张量精确比较；4-token CPU 前向结果保存前后完全一致且有限。

历史文件没有保存 rotary 非持久化 buffer，因此归档时按基础模型配置和当前 Transformers 实现重建，并明确写入归档。这项重载验证不等价于验证历史运行的 PPL；本次没有补报历史 PPL。

删除的是中间保存状态，原有优化任务不能依赖这些缓存直接续跑。完整检查点仍可独立用于模型重载和推理。重新运行需要分层缓存的旧脚本时，应从完整模型重新导出所需层状态或重新生成缓存；原校准激活续跑状态没有保留。

## 重载方式

在仓库根目录使用 `.venv-awq/bin/python`。近期 12 份模型继续使用原接口：

```python
from pruning_methods.OATS.awq.pipeline import load_bundle
model, metadata = load_bundle("runs_awq/llama2_r32_then_r128_20261006_185158/r32/prepare/checkpoint.pt")
```

历史 10 份模型使用新增的独立归档读取器（无需原始基础模型缓存或网络）：

```python
from maintenance.full_model_archive import load_archive
model, metadata = load_archive("old/llama2_oats_24_r32/checkpoint.pt")
```

两种方式默认在 CPU 重建模型。历史归档格式为 `oats-full-archive-v1`；原实验格式仍为 `oats-awq-v1`。历史归档中 `metadata` 记录确切的基础模型版本、补齐规则和重载探针。清理脚本是本次操作的审计记录，不应再次直接运行。

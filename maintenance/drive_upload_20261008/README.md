# OATS 完整模型迁移至 Google Drive

目标：`hopper3/oats_code`

文件夹：https://drive.google.com/drive/folders/1O51iAZNdpGColYAD9haZaTcgku0Fyakq

本次迁移 22 份完整模型，共 298,643,233,198 字节。保留仓库中的相对目录结构，文件名均为 `checkpoint.pt`。上传仅包含清单中的模型和迁移记录，不包含账号凭证或其他本机文件。

- `models_manifest.json`：模型路径、大小、rank、方法、格式、加载函数、MD5 和 SHA256。
- `MD5SUMS` / `SHA256SUMS`：模型文件校验和。
- `verified_receipt.json`：所有模型经 Drive 大小和 MD5 核验通过后，删除本地模型之前写入的记录，包含 Drive 文件 ID。
- `completed_transfer.json`：迁移完成后的记录。
- 本地 `state.json` / `transfer.log` / `rclone.jsonl`：进度、操作日志与错误信息。

只有完整上传并通过远端大小和 MD5 比较的模型才会删除本地副本。各原模型目录保留 `checkpoint.drive.json`，记录远端文件 ID、相对路径和校验和。代码、实验结果、日志、环境与校准输入继续保存在本地。

从仓库根目录恢复全部模型（需要本机现有 rclone 连接）：

```bash
rclone copy gdrive_hdtf: . \
  --drive-root-folder-id 1O51iAZNdpGColYAD9haZaTcgku0Fyakq \
  --files-from-raw maintenance/drive_upload_20261008/model_paths.txt \
  --checksum --immutable --transfers 4
```

恢复后核对 SHA256：

```bash
sha256sum -c maintenance/drive_upload_20261008/SHA256SUMS
```

近期 12 份模型使用 `pruning_methods.OATS.awq.pipeline.load_bundle`；历史 10 份归档使用 `maintenance.full_model_archive.load_archive`。详细格式说明见仓库中 `maintenance/cleanup_20261008/README.md`。

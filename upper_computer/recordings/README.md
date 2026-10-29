# 两版 CNN 的原始录制数据

本分支按 2026-10-29 的指定范围发布完整录制文件，保留原始字节和目录名。`goodone_plus` 是 **S013 四组替换后的当前数据集**；`S013/session_001` 单独保存旧版模型所用的 S013 数据。不要把两个目录一起递归导入同一次训练，否则会重复使用部分 S013 录制。

| 数据目录 | 录制数 | 文件数 | 用途 |
| --- | ---: | ---: | --- |
| `goodone_plus/s011` | 10 | 60 | 两版模型共用 |
| `goodone_plus/s012` | 9 | 54 | 两版模型共用 |
| `goodone_plus/s013` | 10 | 60 | 新版 S013，包含四组 session_003 替换数据 |
| `goodone_plus/s014` | 10 | 60 | 两版模型共用 |
| `goodone_plus/s015` | 10 | 60 | 两版模型共用 |
| `S013/session_001` | 10 | 60 | 旧版 S013，替换前完整十组 |

共 59 个目录、354 个文件、约 2.41 GiB 未压缩数据。每个目录包含 `samples.csv`、`transport.bin`、`frames.jsonl`、`metadata.json`、`events.jsonl`、`trials.jsonl`。使用普通 Git 原始文件，无需 Git LFS 或另行下载数据附件。

## 模型与数据对应关系

| 模型 ID | 冻结实验目录 | 应使用的数据 |
| --- | --- | --- |
| `mid64_temporal3`（两版中的较早版本） | `models/sliding_arch_sweep_20261022_125638_517426/mid64_temporal3` | 当前 goodone_plus 的 s011、s012、s014、s015，加上 `S013/session_001` 的全部 10 组 |
| `mid64_temporal3_20261027`（当前部署新版） | `models/cnn_goodone_current_20261027_214651_552433/mid64_temporal3` | 当前 `goodone_plus` 的全部 49 组 |

两版都使用五人共 49 个录制。架构相同，训练数据中的差异是 S013 替换四组；两次训练、划分和权重产物各自独立。不能把四组录制理解成存在一个已核实的一一配对替换关系。

旧版 S013 中移出的四组：

- `20261021_113627_8c3388`
- `20261021_115441_c9a385`
- `20261021_114145_21a399`
- `20261021_120003_80d5cb`

新版 S013 中加入的四组（原始来源为 `S013/session_003`，本分支在 `goodone_plus/s013` 中提供完整副本）：

- `20261027_205409_a33129`
- `20261027_205949_dfbdb0`
- `20261027_210831_dede40`
- `20261027_211351_d66e33`

S013 其余六组在两个目录中逐文件相同。它们在原始目录中都保留，便于独立取用完整数据集；训练时应只选一个版本。

## 核验与使用

在仓库根目录运行（Python 3.10+，不连接硬件）：

```powershell
python upper_computer/recordings/verify_recordings.py
python upper_computer/board/verify_assets.py
```

数据检查覆盖全部 354 个文件，并将两个冻结实验的训练输入清单映射到本分支数据。两版各 196 个核心文件（每录制的 metadata/samples/events/trials）均已匹配历史 SHA256；S013 替换前后完整文件另与当时替换记录核对，共 120 个文件匹配。数据集文件清单见 [DATASET_MANIFEST.json](DATASET_MANIFEST.json)，模型逐录制映射见 [MODEL_DATASETS.json](MODEL_DATASETS.json)，原始替换证据见 [S013_REPLACEMENT_20261027.json](S013_REPLACEMENT_20261027.json)。

当前新版训练入口为 `upper_computer/run_cnn_goodone_plus_current.ps1`，读取 `recordings/goodone_plus`。这里只发布数据、既有训练脚本和冻结权重，本次未重新训练。旧版的原始实验链包含历史准备产物；原始数据齐备不代表旧版架构扫描脚本可以跳过这些准备阶段直接复跑。复现实验应结合该模型目录中的 protocol、indices、源码快照及输入清单，不改写历史绝对路径和封存哈希。

需要单独组织旧版数据时，可在 `upper_computer` 目录使用以下命令创建新的目录（目标必须尚不存在，保留当前 goodone_plus）：

```powershell
$oldDataset = Join-Path (Get-Location) 'recordings_versions/goodone_plus_20261022'
if (Test-Path -LiteralPath $oldDataset) { throw '目标已存在，请使用新的输出目录。' }
New-Item -ItemType Directory -Path $oldDataset | Out-Null
foreach ($subject in @('s011', 's012', 's014', 's015')) {
    Copy-Item -LiteralPath "recordings/goodone_plus/$subject" -Destination $oldDataset -Recurse
}
Copy-Item -LiteralPath 'recordings/S013/session_001' -Destination (Join-Path $oldDataset 's013') -Recurse
```

离线成绩旧版 405/499、新版 404/499；S013 测试动作不同，不能当作严格相同测试集上的优劣比较。数据筛选使用了已观察的开发评估结果，不是独立盲测；数据发布不改变原实验结论的适用范围。

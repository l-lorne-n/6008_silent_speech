# GitHub 新分支：App、双模型与原始数据

日期：2026-10-29（Asia/Singapore）。仓库 `l-lorne-n/6008_silent_speech`，本次独立分支 `integration-app-data-20261029`，承接 `integration-bt24-models-20261028` 的 `95cce21`。本机发布副本为 `L:\ntu_class\6008\github_publish\6008_silent_speech`。不修改 main；队友原 `firmware/`、`tools/bt24_receiver/` 保留。

## 本次内容

完整上位机源码与启动／依赖文件、训练 experiments 与 semg、源码测试、App 前后端、当前使用的两版 CNN 冻结资产和原有四份预构建 HEX 均随分支提供。同步了 App 顶部静息黄绿状态框（含照片全屏）和可切换场景命令过滤功能。开关默认开启，按当前页面过滤无关命令；没有修改板端解锁、确认或模型分类。

按用户指定上传 `upper_computer/recordings/goodone_plus` 的全部 49 组（294 文件），以及 `upper_computer/recordings/S013/session_001` 的全部 10 组（60 文件）。合计 354 文件，2,586,339,528 字节，约 2.41 GiB。每组六个原始文件全部保留，不仅上传训练使用的 CSV。没有上传其他录制、训练缓存、Python 运行环境、私人照片／音乐或整片固件备份。

## 两版模型与 S013

- 较早版本 `mid64_temporal3`：实验 `sliding_arch_sweep_20261022_125638_517426`，S013 使用 `S013/session_001` 完整十组。
- 新版 `mid64_temporal3_20261027`：实验 `cnn_goodone_current_20261027_214651_552433`，使用当前 goodone_plus，包括 S013 四组替换后的十组。
- 其余 S011、S012、S014、S015 共用当前 goodone_plus；录制数分别为 10、9、10、10。
- 旧 S013 的四组 `113627_8c3388`、`115441_c9a385`、`114145_21a399`、`120003_80d5cb`（日期均 20261021）被移出；加入 20261027 的 `205409_a33129`、`205949_dfbdb0`、`210831_dede40`、`211351_d66e33`，源自 session_003。
- 新四组已在 goodone_plus/s013 提供完整副本，无需再上传整个 session_003。S013 另外六组在两个发布路径中内容相同。

发布分支的 `upper_computer/recordings/README.md` 单独说明完整录制名称、模型对应关系、旧版数据组织方法和复现边界。`MODEL_DATASETS.json` 给出两版各 49 个录制的明确映射；不要把 goodone_plus 和 session_001 同时递归导入一遍训练。

## 核验

- 全部 354 个原始文件逐字节复制并核对 SHA256。
- 两版历史 `inputs_sha256.json` 各 196 个原始核心输入全部匹配；旧版 S013 映射到 session_001，新版映射到当前 goodone_plus。
- S013 替换前后各 60 个完整文件与原替换清单一致。保留原始替换证据，不改写冻结模型清单。
- 两版模型及四份 HEX 的 203 个资产哈希匹配。
- 发布副本 App 26 项软件测试通过；上位机 33 项通过、3 项作者机历史构建路径审计用例排除；两个 JS 语法检查通过。
- 本机 App 已完成场景过滤、开关持久化、全屏提示、静息状态、断线／超时和屏幕尺寸的无头浏览器检查；发布前端与本机字节一致。

仓库根目录执行 `python upper_computer/recordings/verify_recordings.py` 检查数据与两版历史输入，执行 `python upper_computer/board/verify_assets.py` 检查模型和 HEX。完整软件验证记录在 `upper_computer/history/publish_20261029/validation.json`。

本次没有重新训练、连接 STM32 或烧录；BLE 长时间稳定性、UART 故障及真人控制效果仍沿用 2026-10-28 交接中的待验收结论。原始数据齐备不等于旧版多阶段训练链可以跳过准备缓存直接运行；历史源码快照、划分与配置保留供复现使用。

本次提交的精确文件哈希见分支根 `delivery_manifest.json`；最终远端提交与推送确认记录保存在本机 `history/publish_20261029/github_receipt.json`。

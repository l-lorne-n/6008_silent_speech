# S3–S5 中文完整技术报告：CNN 跨人与单人补充版

2026-10-30 更新。保留 S3 信号处理、S4 模型研究、S5 应用集成及既有 LDA 校准内容，新增当前 CNN 五折跨人与五个独立个人模型的实验设计、完整结果、量化表现及评估限制。Word 共 47 页，已更新目录并逐页检查。

- [Word 报告](EE6008_S3-S5_完整技术报告_中文审阅版_CNN跨人与单人补充版.docx)
- [Markdown 阅读版](report.md)
- [CNN 脚本、模型与复现说明](../../upper_computer/model_releases/cnn_20261030/README.md)

FP32 跨人十词分类为 40.30%，连续正确命令率为 37.94%；个人整轮 80/20 十词分类为 92.80%，连续正确命令率为 80.80%。测试人员、训练量和测试成员不同，不应把两者作为同测试集的直接消融对比。个人实验是同录制内整轮留出，不能代表新佩戴泛化。

报告的证据索引保留原工作区 `L:/ntu_class/6008` 下的历史路径，并不表示每份历史中间产物均已发布。此次 CNN 路径映射如下，目录内文件名称保持不变：

| 原工作区目录 | 本仓库发布目录 |
| --- | --- |
| upper_computer/models/cnn_current_loso_20261030 | upper_computer/model_releases/cnn_20261030/cross_subject |
| upper_computer/models/cnn_individual_80_20_20261030 | upper_computer/model_releases/cnn_20261030/individual_80_20 |

训练脚本仍在 `upper_computer/experiments`。部署用两版 CNN 仍按原 `upper_computer/models/README.md` 说明使用；本次十个模型为离线研究资产，未部署到 App 或板端。

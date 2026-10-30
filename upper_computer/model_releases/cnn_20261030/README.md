# 当前 CNN 跨人与个人模型研究发布

2026-10-30 完成两组实验，均使用当前 `mid64_temporal3` 架构与 `recordings/goodone_plus`。本目录包含 **10 个最终模型**及其结果；它们是新离线研究权重，尚未烧录或替换 App 的两版正式模型。

| 实验 | 训练与测试 | FP32 十词分类 | FP32 连续正确命令率 | FP32 严格成功率 |
| --- | --- | ---: | ---: | ---: |
| [跨人五折](cross_subject/results.txt) | 每次另外四人训练，整人留出，无个人校准 | 1006/2496，40.30% | 37.94% | 37.06% |
| [五个独立个人模型](individual_80_20/results.txt) | 各自整轮 80/20，同人的训练和测试 | 464/500，92.80% | 80.80% | 78.80% |

INT8 的三项总体指标分别为：跨人 **40.30%、37.50%、36.66%**；个人 **92.60%、79.20%、77.40%**。个人 S014 的连续正确命令率从 FP32 的 80% 降至 INT8 的 68%，不能只用总体分类率掩盖该变化。

十词分类利用已知动作提示边界，对有效动作窗口平均概率后排除 rest 强制选词。连续正确命令率采用静息确认和连续解码，允许同一试次另有错词或重复；严格成功要求整试次仅一次输出且正确。这里的“连续”是离线按时间回放，不是边用边训练，也不是新权重的板端现场测试。

两组分别测试 2496 与 500 条、训练人数和数据量不同，不能把 92.80% 与 40.30% 的差值视为同测试集上的纯架构收益。个人模型同录制整轮留出仍可能有同次佩戴相关性；跨人数据也已受历史数据整理影响，均不属于新队列独立盲测。

## 文件与模型身份

每组下面的 S011–S015 含义不同：

- `cross_subject/S011/`：模型从 S012–S015 学习，S011 完整留作测试。其余目录同理。
- `individual_80_20/S011/`：模型只用 S011 的训练部分，测试 S011 留出的 100 条。其余目录同理。

各目录的 `mid64_temporal3/` 中包含：

- `model.pt`：最终重训权重、类别顺序、归一化、架构及训练窗口指纹；
- `export/model_fp32.onnx`、`export/model_int8_qdq.onnx`：浮点与量化导出，另保留量化前处理图；
- `export/input_contract.json`：通道、2 kHz、输入形状、滤波、归一化、类别顺序与解码参数；
- `seed_selection.json`、`quantization_selection.json`、`refit_history.json`：种子、轮数和量化选择依据；
- `seeds/*/history.json` 和 `training_summary.json`：三个开发种子的训练历史；
- `test/`：连续概率轨迹、输出事件、逐试次结果及混淆统计。

各人上一级保留 `protocol.json`、`recordings.json`、`dataset_summary.json`、`split_manifest.csv`；各实验根目录保存汇总结果、逐试次分类预测和完成标记。类别顺序必须读取模型自身契约，不手工假定词序。ONNX 输入是标准化后的 `[1,2,1,2800]`；原始 ADC 必须先按契约执行连续因果滤波和训练统计量标准化。

`models.json` 是十个最终模型的索引。`manifest.json` 是此发布包的相对路径 SHA256 清单；`raw_inputs_sha256.json` 按仓库相对路径记录 196 个原始核心文件。原协议及 `original_inputs_sha256.json` 中的本机绝对路径仅为历史证据，不是下载后必须具备的路径。

本包保留最终模型和测试证据，**不包含原始滤波缓存、15 个开发种子的权重或所有窗口训练索引**，因此不是可直接续跑的历史运行目录。复现脚本从仓库附带的原始录制重新生成这些内容。不要把本目录复制到 `models/` 冒充完整运行。

## 校验下载

在仓库根目录运行以下命令，只校验文件时仅需 Python 标准库：

```powershell
python upper_computer/model_releases/cnn_20261030/verify.py
python upper_computer/model_releases/cnn_20261030/verify.py --data
```

安装训练依赖后，可检查十个 PyTorch 检查点与 FP32 ONNX 数值一致性、INT8 图结构、人员与训练分区，并从保存的概率轨迹重新解码验证连续评分：

```powershell
upper_computer/.runtime/Scripts/python.exe -X utf8 upper_computer/model_releases/cnn_20261030/verify.py --data --models
```

`verification.json` 保存发布前实测检查结果。数值一致性测试采用固定随机标准化输入，确认导出执行一致性；它不是额外的识别准确率实验。

## 从原始数据复现

使用本分支已有 `goodone_plus`，保留它当前 S013 四组替换后的版本。新电脑先准备 Windows x64 Python 3.12 环境。以下命令从 `upper_computer` 目录执行；已有项目 `.runtime` 时不必重复建立虚拟环境：

```powershell
python -m venv .runtime
./.runtime/Scripts/python.exe -m pip install -r requirements.txt
./.runtime/Scripts/python.exe -m pip install -r requirements-cnn.txt
./.runtime/Scripts/python.exe -m pip install -r requirements-onnx.txt
```

本实验保存的关键版本为 torch 2.14.1+cpu、numpy 2.5.3、onnx 1.17.0。代码也兼容项目原有 `.cnn-deps` 与 `.onnx-deps` 布局；若这些目录已经存在，应核对实际导入版本。跨机器运行时间、底层计算与最终小数不保证逐位一致，划分、选参规则和模型契约应保持一致。

跨人实验双击上两级的 [`run_cnn_cross_subject.cmd`](../../run_cnn_cross_subject.cmd)，或：

```powershell
./.runtime/Scripts/python.exe -u -X utf8 experiments/run_cnn_current_loso.py --jobs 2 --threads 4
```

个人实验双击 [`run_cnn_individual_80_20.cmd`](../../run_cnn_individual_80_20.cmd)，或：

```powershell
./.runtime/Scripts/python.exe -u -X utf8 experiments/run_cnn_individual_80_20.py --jobs 2 --threads 4
```

两组均每人/每折三个随机种子，由训练侧整轮验证选择种子和轮数，最终合并训练与验证从头重训。个人最终训练数为 400、400、400、396、400，各测 100；跨人最终源训练为 1996、1996、1996、2000、1996，测试为 500、500、500、496、500。更详细的开发/验证数量见协议文件。

两个 CMD 使用不同的本地输出目录，并支持完整准备后的中断续跑。跨人 CLI 的自选输出续跑需要 `--output <原目录> --resume`；个人 CLI 在同一输出目录自动续跑。准备阶段中断时使用新的输出目录；未完成种子从头训练，已完成阶段按哈希核验后复用。训练可能耗时较长，`--jobs 1` 可减少并行任务。

`continue_cnn_current_loso.py`、`run_cnn_loso_remaining.cmd` 及早期续跑说明仅归档原本机阶段流程。原运行曾按要求先测 S011/S012 再完成另外三折，协议修订明确记录此事；新下载复现请用上述完整跨人入口，其默认是全部模型冻结后再统一测试，模型训练与评分规则不变。

完整报告见仓库 [`reports/20261030_S3_S5_CNN`](../../../reports/20261030_S3_S5_CNN)。

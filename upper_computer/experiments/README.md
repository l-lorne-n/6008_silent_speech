# 小型CNN离线实验

与主界面分离，不更改采集、LDA或在线识别流程。模型保存到新的`models/cnn_*`目录，不能加载到当前LDA模型窗口。

用已有`.runtime`运行，将可选CPU版PyTorch装入独立目录：

```powershell
.runtime/Scripts/python.exe -m pip install --target .cnn-deps -r requirements-cnn.txt
.runtime/Scripts/python.exe experiments/train_small_cnn.py --baseline models/train_20261010_173540_db0bb3
```

脚本支持十词、双通道的已保存LDA基线，不再固定录制数和试次数。直接传入对应训练目录：例如S05 session_014使用`--baseline models/train_20261010_204942_cdf953`。原LDA的试次筛选及测试划分保持不变；每份包含训练数据的录制须至少有两轮训练数据，从中固定随机抽一整轮验证。检查试次指纹唯一、整轮不跨集合、训练与验证均覆盖所有类别；不满足则报错。

协议：冻结LDA测试试次，从原训练试次中每份录制抽一整轮验证。固定架构、三个预定随机种子，按验证交叉熵选检查点与轮数。然后从头用原LDA全部训练样本、选定种子和轮数重新训练，最后在原测试集评估一次。禁止看测试结果后改网络、增强或种子。首次11组实验为330/110/110，最终440训练/110测试；其他数据集按实际整轮大小记录数量，不强制每人或每批次比例相同。

输入：两路原始ADC按LDA同样的因果滤波、准备期预热及动作前后各150 ms裁剪处理，右侧补零到2048点，不拉伸时间；逐试次原始指纹和重新算出的LDA特征均与基线核对。各通道均值/标准差仅在当前训练部分的非补零点拟合，验证/测试沿用；最后重训时在原LDA全部训练试次上重估。

网络：Conv1d 2→16→32→64，BatchNorm、ReLU、平均池化、6段自适应平均池化、Dropout及线性十分类，共23498参数。AdamW，lr=0.001，weight_decay=0.001，batch=32，最多120轮、耐心20轮；仅训练时使用两通道共同±40点平移（空位补零）及各通道0.9–1.1幅度缩放。CPU四线程、确定性算法。

每次输出保存protocol.json、脚本快照、split_manifest.csv、验证历史、最终重训历史、model.pt、逐次预测、报告及中文说明。model.pt包含权重、归一化、类别顺序及预处理协议。加载使用torch.load(..., weights_only=True)，配合本脚本SmallCNN类；模型重载后已做数值一致性检查。

图表脚本`plot_small_cnn.py <run目录>`需要额外matplotlib，仅读取结果，不影响模型选择。原始数据文件与基线报告在执行前后核对哈希，原模型不覆盖。

这是已见人员/已见录制内的提示窗口分类，开发数据已在此前实验中被查看过。不能据此声称跨人泛化、自动活动切段准确率或实时端到端性能。应在模型方案冻结后补充新录制的独立验证。

## 小型 LSTM 对照

`train_small_lstm.py` 沿用上述预处理、增强、优化器、预定种子及验证选择流程，另外逐试次核对指定 CNN 的训练/验证/测试划分完全相同。不会修改原 CNN 脚本、权重或录制。

```powershell
.runtime/Scripts/python.exe experiments/train_small_lstm.py --baseline models/train_20261012_211031_2fa041 --cnn-reference models/cnn_20261013_101157_1f3069
```

输入仍为 2×2048，每 16 个连续采样点无损重排成一个时间步，得到 128×32；没有降采样、包络提取或卷积前端。单层单向 LSTM、64 个隐藏单元，输出按时间自适应平均池化至 6 段、Dropout 0.35、线性十分类，共 28938 参数。保留与 CNN 相同的补零尾部；这是分帧波形 LSTM 的一个固定架构实验，不代表所有 LSTM 方案。

结果保存到新的 `models/lstm_*` 目录，模型 schema 为 `semg_small_lstm_v1`。包含 CNN/LDA 成绩和逐试次成对正确数，测试集不参与选择。绘图运行 `experiments/plot_small_lstm.py <run目录>`，需要 matplotlib。加载 `model.pt` 时使用本脚本的 `SmallLSTM`，不要使用 CNN 类或 LDA GUI。

## 单向跨受试者测试

```powershell
.runtime/Scripts/python.exe experiments/run_cross_subject.py --train-session recordings/S04/session_003 --test-session recordings/S05/session_014
```

脚本默认只运行 LDA 基线。读取两份 session 下全部录制，核对人员不同、动作条件一致，再按现有质量规则提取。源受试者全部有效动作用于训练 LDA，目标受试者全部有效动作只用于测试，不参与标准化或参数选择。

只有显式添加 `--with-cnn` 才额外运行 CNN：从源受试者各录制抽一整轮验证，验证选择后从头用全部源数据重训。目标数据不参与均值/标准差拟合、BatchNorm 更新、早停或参数选择。沿用现有 SmallCNN、三个初始化、优化器及增强，不加载历史混合模型的权重。

LDA 保存至新的 `models/train_*`，CNN 保存至新的 `models/cnn_cross_*`；CNN 报告记录双方人员与 `evaluation_scope=cross_subject`，并校验基线报告和权重哈希。`plot_small_cnn.py` 支持该报告，只绘制有测试数据的人员。

这是一个方向的跨人测试，不是完整五人留一验证。两份数据已参与历史开发，须说明这是探索性迁移评估；不应根据目标人的测试成绩继续调参。更改方向请重新明确训练/测试目录并生成新的模型，不能将双向结果冒充五折 LOSO。

## goodone 四人 LDA 留人测试

```powershell
.runtime/Scripts/python.exe experiments/train_lda_loso.py
```

默认带通为 20–400 Hz，50 Hz 陷波 Q=30。使用 `--bandpass-high 450` 可运行 20–450 Hz 对照，其他设置保持一致；保存为独立的 `notch50_bp450` 滤波变体，需要包含该变体的新版 `semg/lda.py` 加载。已有 `notch50` 模型及默认滤波不变。

使用 `--bandpass-high 400 --feature-set rms_mav_zcr_wl` 可比较 RMS、MAV、ZCR、波形长度四种特征。仍为双通道、每通道 6 段，共 48 维。RMS 为均方根，MAV 为平均绝对值，ZCR 为严格异号相邻点对数除以 N−1（零阈值，恰为零不计），波形长度为相邻差绝对值之和；四者均不取对数。默认 `--feature-set legacy` 保留历史特征，模型保存完整特征配置。新版界面也可单独选择新增四种特征。

默认只递归读取 `recordings/goodone/s1`、`s2`、`s4`、`s5`，不读取 s3。固定当前 48 维双通道特征，数据只提取一次；每折三个人训练、剩下一人全部测试，四个人各留出一次。各折 StandardScaler 和 lsqr/auto 收缩 LDA 独立拟合，仅使用训练人员，不调参或按结果筛数据。

输出 `models/lda_loso_goodone_*`，包含各折可加载的 LDA 模型、全部逐次预测、来源哈希、排除原因、每人结果、人员等权平均及全部测试动作的合并结果。校验训练/测试人员隔离、每个有效动作恰好测试一次、各折模型重载、输入文件不变。该结果只代表选定子集上的四折留人评估，不能冒充完整五人测试或实时识别成绩。

## goodone 四人小型 CNN 留人测试

```powershell
.runtime/Scripts/python.exe experiments/train_cnn_loso.py --check-only
.runtime/Scripts/python.exe experiments/train_cnn_loso.py
```

默认使用 `models/lda_loso_goodone_20261014_130607_0c55eb` 中冻结的四折划分，也可通过 `--baseline` 指定同协议的 LDA 汇总目录。预检确保 S01/S02/S04/S05、20–400 Hz、双通道十分类、训练与测试人员互斥，各动作恰好测试一次。

四折依次运行既有 `SmallCNN`：2×2048 输入、16→32→64、23498 参数。每折从三个训练人员的各录制中抽取一整轮验证，用三个预定种子的验证损失选择种子及轮数，再从头用该折全部源人员数据重训。目标人员不参与标准化、BatchNorm 更新、早停或选择。参数与增强沿用上面的固定 CNN 方案，不根据四折测试表现调整。

输出独立的 `models/cnn_loso_goodone_*`，含每折训练日志、协议、训练曲线、模型、逐次预测以及总报告。汇总前再次核对测试身份和原始哈希，复算同一测试集 LDA 结果。`--check-only` 只验证协议，不训练或创建输出。内部调用 `train_small_cnn.py --output <新目录>`，已有输出目录会被拒绝覆盖。绘图使用 `experiments/plot_cnn_loso.py <汇总目录>`，需 matplotlib。

CNN 输入滤波波形，不使用 RMS/MAV 等人工特征作为网络输入；重算 LDA 特征仅用于验证波形预处理和数据身份。模型尚未转换或部署到 STM32，桌面推理计时不能视为板上速度。

若进程在验证选择完成后、最终模型保存前中断，可用 `train_cnn_loso.py --resume <原汇总目录>` 恢复：完整折直接核对并复用；未完成折须已有全部三个种子的选择历史和冻结参数，并与原协议完全匹配，再从同一种子重新开始最终重训。日志追加保存，原脚本快照保留。该恢复入口不支持任意训练轮次的优化器断点续训，也拒绝重训已保存最终模型但缺报告的异常状态。

## LDA 整动作归一化跨人对照

```powershell
.runtime/Scripts/python.exe experiments/train_lda_normalization_loso.py --check-only
.runtime/Scripts/python.exe experiments/train_lda_normalization_loso.py
```

默认冻结原四人 LDA 基线的1998次动作、39份录制及质量排除，保持20–400 Hz、50 Hz陷波、六段原四特征及训练人员专用StandardScaler/LDA。比较原方案、整动作两路分别Z-score、整动作每通道去均值后两路共同尺度三种方法。共同尺度为两通道去均值样本的整体RMS；标准差均用ddof=0，尺度下限1e-6 ADC计数。归一化在滤波和裁边之后、分段之前，不包含准备期或补零，也不使用其他目标动作或目标标签。`--check-only` 验证变换的不变性、通道比例和零输入行为。

输出 `models/lda_normalization_loso_*`，保存特征、归一化统计、各折独立实验模型及逐次预测；核对源文件、试次指纹、原始特征和原基线全部预测。模型schema为 `semg_lda_trial_normalization_experiment_v1`，不能直接用于现有GUI。没有修改正式提取流程，也不运行CNN。

2026-10-14结果：[归一化对照汇总](../models/lda_normalization_loso_20261014_151207_9550f6/结果汇总.md)。四人平均依次为20.68%、14.72%、17.82%，本次两种逐动作归一化均未改善整体结果。12个保存模型的独立核验见该目录 `saved_models_audit.json`。这些是已查看子集上的探索性四折结果，不证明所有归一化方式无效；当前继续保留原方案。

## LDA 通道消融跨人对照

```powershell
.runtime/Scripts/python.exe experiments/train_lda_channels_loso.py
```

重新提取并核对原四人基线的1998次动作与特征，在相同四折留人划分下比较喉咙CH0（24维）、脸部CH1（24维）和双通道（48维）。每组均保留原双通道质量规则排除的2次动作，即使只使用单路也不重新放入这些动作，避免比较不同样本集。滤波、六段原四特征与LDA设置不变，不做额外逐动作归一化。

输出独立的 `models/lda_channels_loso_*`，包含各折模型、预测、数据审计与汇总；schema为 `semg_lda_channel_ablation_v1`，不直接用于GUI。2026-10-14 [结果汇总](../models/lda_channels_loso_20261014_151634_22ec18/结果汇总.md)：喉咙18.77%、脸部15.52%、双通道20.68%。双通道全部预测复现原基线，保存模型和训练侧标准化核验通过，另有 `independent_prediction_audit.json`。没有一致证据支持删除某一路；不能把不同测试人员各自最好的通道组合成部署成绩。仍属于已查看数据上的探索性评估。

## LDA 数据量与训练人数学习曲线

```powershell
.runtime/Scripts/python.exe experiments/train_lda_learning_curves.py
```

读取上面通道对照保存且已验证的原始48维特征，逐条核对原LDA报告并检查原始录制哈希。固定双通道原特征及LDA设置，20个种子20261014–20261033，每次完整留出一人测试。实验一固定另外三人，用每人10/25/50个完整轮次的嵌套抽样；轮次中原来被排除的两次削顶动作不恢复，因此为计划次数而非严格等量。实验二固定300次训练动作，1/2/3个训练人分别每人每词30/15/10次，遍历全部训练人员组合。实验二仅从完整十词轮次抽取训练数据，S05两个不完整轮次的18次有效动作不作该实验训练候选，但测试全部保留。

先对训练人员组合等权，再对四名测试人员等权，报告20次抽样的均值和标准差。这些重复不等于20批独立人员；50轮档已用全部源数据，不重复拟合同一训练集。共800个评估记录、724个独立拟合，全部保存模型、索引清单和预测。输出 `models/lda_learning_curves_*`；模型保存在独立NPZ中，未接入GUI。

2026-10-14 [结果汇总](../models/lda_learning_curves_20261014_155344_e64856/结果汇总.md)：10/25/50次档为19.98%/20.94%/20.68%；固定300次动作，1/2/3人档为14.63%/17.78%/20.36%。当前数据更支持优先增加训练人员；不能外推50→100次或4→8人的具体效果。800个保存模型和独立抽样/汇总核验均通过。绘图脚本 `plot_lda_learning_curves.py <结果目录>` 使用matplotlib，只读已有报告，不训练或选模型；本机可用 `D:/python/python.exe` 运行。

## 四人混合 CNN：每人 80% 训练、20% 测试

```powershell
.runtime/Scripts/python.exe experiments/train_cnn_mixed_goodone.py
```

用户指定 S01/S02/S04/S05 的已见人员混合评估。重新提取原 LOSO 基线同一批39份录制，核对1998次有效动作、48维审计特征和原始文件哈希；保留原有两次质量排除。固定种子20261010，各录制内按完整轮次80/20划分，每人40轮训练、10轮测试。原始动作及其波形片段不会跨集合。建立同一划分的 LDA 对照，再调用现有 `train_small_cnn.py`，沿用23498参数、双通道、20–400 Hz、三次预定初始化和训练侧标准化。

从80%训练部分的各录制抽一整轮作为开发验证，仅按验证损失选择种子和训练轮数；然后从头用完整80%重训，最后评估20%测试集。输出独立 `models/cnn_mixed_goodone_*`，包含 `protocol.json`、LDA子目录、`cnn/model.pt`、逐次预测、训练日志及报告。既有数据曾在此前实验中被查看，结果属于探索性留出评估；这里测量的是已见人员、已见录制批次的新动作表现，不能替代跨人或跨重贴电极测试。

任务书原文的第8周明确写了四训一测轮换和跨人基线目标；第9周要求DL训练并与传统基线比较，第11周要求留出评估及五人演示，但没有在CNN里另行重申四训一测。混合评估可回答已见用户识别问题，不意味着任务书的跨人目标或五人覆盖要求取消。当前四人实验不宣称完整验收通过。

2026-10-14结果：[实验交接](../history/2026-10-14_四人混合CNN_80比20.txt)。1599训练、399测试；CNN **323/399 = 80.95%**，宏F1 0.8072，同划分LDA **203/399 = 50.88%**。CNN逐人S01/S02/S04/S05为83.00%/82.00%/80.00%/78.79%；只凭验证集选种子20261011及119轮，最终用完整训练集重训。结果保存在 `models/cnn_mixed_goodone_20261014_161620_3d5ba7`，`saved_outputs_audit.json` 已通过保存模型成员、划分、逐次预测和汇总核验，图表在 `cnn/cnn_vs_lda.png`。

五人版本（用户随后明确要求纳入 S03）：

```powershell
.runtime/Scripts/python.exe experiments/train_cnn_mixed_goodone.py --include-s3
```

此选项扫描 `recordings/goodone/s1` 至 `s5` 全部录制，检查人员和静默采集条件，保留相同质量规则、固定种子、按整轮80/20划分、CNN结构与验证选择流程。输出独立 `models/cnn_mixed_5p_goodone_*`。默认不加选项仍为原四人版本。五人重新应用同一划分算法；加入S03会消耗额外随机数，后续人员的测试轮次可能变化，因此四人和五人的总体准确率不是只改变训练人数的受控对照。

五人结果：[实验交接](../history/2026-10-14_五人混合CNN_80比20.txt)。1999训练、499测试，CNN **377/499 = 75.55%**，宏F1 0.7537；同划分LDA **215/499 = 43.09%**。S01/S02/S03/S04/S05分别为86.00%/76.00%/56.00%/77.00%/82.83%。验证集选种子20261010及111轮，最终完整训练集重训后一次测试。目录 `models/cnn_mixed_5p_goodone_20261014_163748_de5b49`；`verify_cnn_mixed_outputs.py <实验目录>` 的划分、模型成员及预测汇总审计通过。

## 2026-10-17：新三人2 kHz数据

发布附件解压后可运行：

```powershell
.runtime/Scripts/python.exe experiments/train_lda_loso.py --root recordings --people S011 S012 S015 --bandpass-high 500
```

该脚本支持任意至少两位指定受试者；此处每折两人训练、一人测试，共三折。数据为三人原始通道顺序，各500次完整动作；S012已删除的中止录制不在附件中。`--exclude-recording` 只在目标录制确实存在且需要排除时使用。输出为独立 `models/lda_loso_3p_*`，含来源哈希、各折模型、预测和汇总。正常按质量规则跳过已作废动作。

`audit_*`、`compare_s0*_channel_swap.py` 是历史探索脚本，部分依赖本机历史模型或未公开更正备份，不属于上位机启动步骤。信号相似性不能单独证明解剖通道位置。

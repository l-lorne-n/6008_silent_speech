# 当前 CNN 跨人测试：继续运行剩余三折

2026-10-30 已完成 S011、S012 两折的全部训练、FP32/INT8 测试和结果复核。S013、S014 曾在自动队列切换时启动少量训练，已按用户要求停止；没有完成种子。S015 尚未训练。

## 怎么启动

双击同目录的 **`run_cnn_loso_remaining.cmd`**。无需插 STM32，也无需打开 App。

它使用本项目已有的 `.runtime` Python 环境，保留已经完成的 S011/S012，依次安排 S013、S014、S015 的训练。默认同时跑两折，每折使用 4 个 PyTorch CPU 线程。每折仍然是三个种子、按源人员验证集选择种子和轮数、最终合并源人员训练与验证数据重训，然后做浮点版和 INT8 版测试。

也可以在 PowerShell 中运行：

```powershell
Set-Location L:\ntu_class\6008\upper_computer
& .\.runtime\Scripts\python.exe -u -X utf8 .\experiments\continue_cnn_current_loso.py --jobs 2 --threads 4
```

若希望同时只训练一折，将 `--jobs 2` 改为 `--jobs 1`。不需要修改种子、数据或模型代码。

## 查看进度与结果

命令窗口显示每折启动、完成和最终结果。详细逐轮日志在：

- `models/cnn_current_loso_20261030/S013/continue_fit.log`
- `models/cnn_current_loso_20261030/S014/continue_fit.log`
- `models/cnn_current_loso_20261030/S015/continue_fit.log`

可另开 PowerShell 实时看日志：

```powershell
Get-Content L:\ntu_class\6008\upper_computer\models\cnn_current_loso_20261030\S013\continue_fit.log -Tail 10 -Wait
```

这个查看日志的窗口按 Ctrl+C 只会停止查看。查看训练阶段状态可运行：

```powershell
& L:\ntu_class\6008\upper_computer\.runtime\Scripts\python.exe -X utf8 L:\ntu_class\6008\upper_computer\experiments\continue_cnn_current_loso.py --phase status
```

当前两折结果是 `models/cnn_current_loso_20261030/partial_results.txt`，**不是五折总体结果**。

剩余三折完成后，同一目录会自动生成：

- `results.txt`：完整五折中文汇总；
- `comparison.json`、`comparison.csv`：连续命令指标与各人的结果；
- `ten_word_comparison.json`：按动作提示分段的十词分类准确率；
- `fp32_trials.csv`、`int8_trials.csv`：逐试次连续命令结果；
- `fp32_ten_word_trials.csv`、`int8_ten_word_trials.csv`：逐试次十词分类预测；
- `completed.json`：只有完整五折完成并通过复核后才生成，`full_five_fold` 应为 `true`。

## 中断后继续

关闭运行窗口或中断后，重新双击同一 CMD 即可。已经完成且哈希核对通过的种子、重训、导出和测试阶段直接复用；中断在一半的种子会从第一轮重新训练，不会从不完整检查点接着算。

保留当前实验目录、原始 goodone_plus 和实验依赖代码。不要同时打开两个续跑任务，也不要为了重新运行而删除实验目录。若输入文件被修改，哈希检查会报错，避免混用不同实验条件。

本次继续运行请使用新的 `continue_cnn_current_loso.py` 或 CMD。原始 `run_cnn_current_loso.py` 的全流程采用“五折全部冻结后统一测试”的策略；用户后来要求先完成当前两折，因此新的入口记录了这项阶段调整，而训练与评分规则保持不变。

## 两种分数的意思

十词分类准确率：借助已有动作提示边界，汇总动作窗口的平均概率，排除 rest 后在十个词中选一个。它是离线分类诊断，不代表在线触发成功率。

连续正确命令率：沿用当前静息确认和命令解码规则，响应时段内至少输出过一次正确词。同一试次可能还出现错词或重复。

严格成功率：整个试次上下文只输出一次，并且该输出在响应时段内、词也正确。

本实验没有目标人员校准，也不使用目标人的均值方差。量化流程中名为 calibration 的 550 个窗口全部来自其他四人，只用于估计 INT8 激活范围。运行不会改动现有 App、原模型或板端程序。

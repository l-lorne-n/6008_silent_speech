"""Plot saved four-subject CNN results; never used for model selection."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1]).resolve()
r = json.loads((root/'report.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False, 'font.size':11})
fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
labels = [f['subject'] for f in r['folds']]+['四人平均']
x = np.arange(len(labels))
old = [f['lda_metrics']['accuracy']*100 for f in r['folds']]+[r['lda_subject_mean_accuracy']*100]
new = [f['cnn_metrics']['accuracy']*100 for f in r['folds']]+[r['subject_mean_accuracy']*100]
for offset, values, label, color in [(-.19,old,'LDA','#8597a3'),(.19,new,'小型 CNN（2.35 万参数）','#207c93')]:
    bars=axes[0].bar(x+offset,values,.36,label=label,color=color)
    axes[0].bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=4,fontsize=9)
axes[0].axhline(50,color='#af5e45',linestyle='--',label='跨人目标准确率 50%')
axes[0].set(xticks=x,xticklabels=labels,ylabel='测试准确率 %',ylim=(0,max(65,max(new+old)+16)),title='相同数据、相同人员划分')
axes[0].legend(loc='upper right',fontsize=9)
axes[0].set_axisbelow(True)
axes[0].grid(axis='y',alpha=.15)
m=np.array(r['pooled_metrics']['confusion_matrix'])
classes=r['pooled_metrics']['row_labels']
axes[1].imshow(m,cmap='Blues')
axes[1].set(xticks=range(len(classes)),xticklabels=classes,yticks=range(len(classes)),yticklabels=classes,
            xlabel='预测词语',ylabel='真实词语',title=f"CNN 合并混淆矩阵：{r['pooled_metrics']['correct']}/{r['pooled_metrics']['total']}")
plt.setp(axes[1].get_xticklabels(),rotation=50,ha='right')
for i in range(len(classes)):
    for j in range(len(classes)):
        axes[1].text(j,i,str(m[i,j]),ha='center',va='center',fontsize=8,color='white' if m[i,j]>m.max()*.55 else '#222222')
fig.suptitle('goodone 四人跨人测试：每折三人训练、一人测试；20–400 Hz；排除 S03')
fig.savefig(root/'cnn_vs_lda_loso.png',dpi=160)
plt.close(fig)
print(root/'cnn_vs_lda_loso.png')

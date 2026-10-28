"""Plot saved CNN experiment outputs; never trains or selects a model."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
folder=args.run
report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
history=json.loads((folder/'selection_history.json').read_text(encoding='utf-8'))
baseline=json.loads((Path(report['protocol']['baseline'])/'report.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
for item in history:
    epochs=item['epochs'];label=str(item['seed'])
    axes[0,0].plot([v['epoch'] for v in epochs],[v['validation_loss'] for v in epochs],label=label)
    axes[0,1].plot([v['epoch'] for v in epochs],[v['validation_accuracy']*100 for v in epochs],label=label)
selected=report['selected']
axes[0,0].scatter([selected['epoch']],[selected['validation_loss']],s=75,color='black',marker='*',zorder=10,label='验证集选中')
counts=report['protocol']['split_counts']
axes[0,0].set(title=f'{counts["train"]}次训练 / {counts["validation"]}次验证：三次预定初始化',xlabel='训练轮数',ylabel='验证交叉熵')
axes[0,1].set(title='验证准确率（用于开发，非最终测试成绩）',xlabel='训练轮数',ylabel='准确率 %')
for ax in axes[0]:ax.legend();ax.grid(alpha=.2)
tested_groups=[g for g in report['group_metrics']['subject'] if g['test_total']>0]
subjects=['总体']+[g['subject'] for g in tested_groups]
cnn=[report['metrics']['accuracy']]+[g['metrics']['accuracy'] for g in tested_groups]
old={g['subject']:g['metrics']['accuracy'] for g in baseline['group_metrics']['subject']}
lda_values=[baseline['metrics']['accuracy']]+[old[s] for s in subjects[1:]]
xx=np.arange(len(subjects))
for shift,values,color,label in [(-.18,lda_values,'#6a879c','LDA'),(.18,cnn,'#208467','小型CNN')]:
    bars=axes[1,0].bar(xx+shift,np.array(values)*100,width=.35,color=color,label=label)
    axes[1,0].bar_label(bars,fmt='%.1f%%',padding=3)
axes[1,0].set(xticks=xx,xticklabels=subjects,ylim=(0,100),ylabel='准确率 %',title=f'同一{report["final_train_trials"]}次训练 / 同一{report["test_trials"]}次测试')
axes[1,0].legend(loc='upper center',ncol=2)
matrix=np.array(report['metrics']['confusion_matrix'])
axes[1,1].imshow(matrix,cmap='Blues')
classes=report['protocol']['classes']
axes[1,1].set(xticks=range(10),xticklabels=classes,yticks=range(10),yticklabels=classes,title='CNN测试混淆矩阵',xlabel='预测',ylabel='真实')
plt.setp(axes[1,1].get_xticklabels(),rotation=55,ha='right')
for i in range(10):
    for j in range(10):
        axes[1,1].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>matrix.max()/2 else '#222222')
if report['protocol'].get('evaluation_scope')=='cross_subject':
    fig.suptitle('跨受试者测试：'+','.join(report['protocol']['train_subjects'])+' 训练 → '+','.join(report['protocol']['test_subjects'])+' 测试')
fig.savefig(folder/'cnn_vs_lda.png',dpi=150)
print(folder/'cnn_vs_lda.png')

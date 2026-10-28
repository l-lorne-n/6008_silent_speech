"""Plot existing learning-curve results; performs no fitting or selection."""
import json
import os
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
os.environ.setdefault('MPLCONFIGDIR',str(root/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=json.loads((root/'report.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for ax,experiment,title,xlabel in zip(axes,['volume','diversity'],
        ['固定三名训练人：增加每人重复次数','固定300次训练动作：增加训练人数'],
        ['每人每词计划次数','训练人数']):
    points=[s for s in r['summary'] if s['experiment']==experiment]
    levels=[s['level'] for s in points]
    for person,color in zip(r['protocol']['subjects'],['#6a85b6','#db9669','#77a694','#ae8fae']):
        ax.plot(levels,[s['by_subject'][person]['mean']*100 for s in points],marker='.',color=color,alpha=.8,label=person)
    means=[s['mean']*100 for s in points]
    ax.errorbar(levels,means,yerr=[s['sampling_sd']*100 for s in points],color='#172d48',
        linewidth=2.6,marker='o',capsize=5,label='四人等权平均 ± 抽样标准差')
    for x,y in zip(levels,means):ax.annotate(f'{y:.2f}%',(x,y),xytext=(0,12),textcoords='offset points',ha='center',fontweight='bold',color='#172d48')
    ax.axhline(10,color='#aaa',ls=':',linewidth=1)
    ax.set(title=title,xlabel=xlabel,ylabel='跨人测试准确率 (%)',xticks=levels,ylim=(0,35))
    ax.grid(axis='y',alpha=.15)
    ax.spines[['top','right']].set_visible(False)
axes[0].legend(loc='lower right',fontsize=8)
fig.suptitle('LDA 跨人学习曲线｜相同四人测试集，20次预定抽样',fontsize=15,fontweight='bold')
fig.supxlabel('误差棒仅表示固定四人数据上的抽样波动；50次档已用全部训练数据。不能据此外推新增人员或100次/词的成绩。',fontsize=9)
fig.savefig(root/'learning_curves.png',dpi=170)
fig.savefig(root/'learning_curves.svg')
plt.close(fig)
print(root/'learning_curves.png')

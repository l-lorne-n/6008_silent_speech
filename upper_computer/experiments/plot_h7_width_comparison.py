"""Plot saved comparisons only; never trains or changes model selection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run',type=Path);folder=p.parse_args().run
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
r=read(folder/'comparison.json');history=read(folder/'cnn/selection_history.json')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
subjects=['总体']+r['protocol']['subjects'];xx=np.arange(len(subjects))
colors=['#788699','#b5c3d1','#16816f','#62b9a9']
for j,(key,label) in enumerate([('small_fp32','原 FP32'),('small_int8','原 INT8'),('wide_fp32','新 FP32'),('wide_int8','新 INT8')]):
    model,dtype=key.split('_');values=[r[model][dtype]['accuracy']]
    values.extend(r['subjects'][key][s]['correct']/r['subjects'][key][s]['total'] for s in subjects[1:])
    bars=axes[0,0].bar(xx+(j-1.5)*.20,np.array(values)*100,width=.19,color=colors[j],label=label)
    axes[0,0].bar_label(bars,fmt='%.1f',fontsize=7,padding=2 if j%2==0 else 9)
axes[0,0].set(xticks=xx,xticklabels=subjects,ylim=(0,104),ylabel='准确率 %',title='同一499条测试：原16/32/64与新32/64/128')
axes[0,0].legend(ncol=4,loc='lower center',fontsize=8)
for h in history:
    axes[0,1].plot([e['epoch'] for e in h['epochs']],[e['validation_loss'] for e in h['epochs']],label=str(h['seed']))
chosen=r['wide']['selected'];axes[0,1].scatter(chosen['epoch'],chosen['validation_loss'],marker='*',s=110,color='#111111',label='验证选中',zorder=5)
axes[0,1].axhline(r['small']['selected']['validation_loss'],color='#778899',linestyle='--',label='原模型最佳验证损失')
axes[0,1].set(title='新模型三次初始化：验证交叉熵',xlabel='训练轮数',ylabel='验证损失')
axes[0,1].grid(alpha=.2);axes[0,1].legend(fontsize=8)
matrix=np.array(r['wide']['int8']['confusion_matrix']);classes=r['wide']['int8']['column_labels']
axes[1,0].imshow(matrix,cmap='Blues')
axes[1,0].set(title='新模型INT8测试混淆矩阵',xticks=range(10),yticks=range(10),xticklabels=classes,yticklabels=classes,xlabel='预测',ylabel='真实')
plt.setp(axes[1,0].get_xticklabels(),rotation=45,ha='right')
for i in range(10):
    for j in range(10):axes[1,0].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>matrix.max()/2 else '#222222')
ref=read(Path(r['protocol']['comparison_reference'])/'cnn_6144/deployment_calibrated/resource_estimate.json')
new=r['resources'];labels=['权重/偏置\nKiB','张量清单\nKiB','含256KiB预留\nKiB']
fields=['int8_weight_plus_int32_bias_bytes','tensor_inventory_int8_bytes','int8_tensor_inventory_plus_reserve_bytes']
for offset,data,color,label in [(-.18,ref,'#788699','原16/32/64'),(.18,new,'#16816f','新32/64/128')]:
    bars=axes[1,1].bar(np.arange(3)+offset,[data[k]/1024 for k in fields],width=.35,color=color,label=label)
    axes[1,1].bar_label(bars,fmt='%.1f',padding=3)
axes[1,1].set(xticks=range(3),xticklabels=labels,ylim=(0,800),title='资源规划：非实测RAM，权重按Flash存储',ylabel='KiB')
axes[1,1].legend(loc='upper left')
fig.suptitle('五人混合CNN加宽实验｜整轮隔离；测试集曾被查看；未部署STM32',fontsize=14)
fig.savefig(folder/'comparison.png',dpi=160)
print(folder/'comparison.png')

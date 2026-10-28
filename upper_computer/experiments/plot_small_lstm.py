"""Plot saved LSTM/CNN/LDA comparisons without training or model selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    folder = parser.parse_args().run
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    history = json.loads((folder / 'selection_history.json').read_text(encoding='utf-8'))
    baseline = json.loads((Path(report['protocol']['baseline']) / 'report.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for item in history:
        epochs = item['epochs']
        for ax, key in zip(axes[0], ['validation_loss', 'validation_accuracy']):
            scale = 100 if key == 'validation_accuracy' else 1
            ax.plot([e['epoch'] for e in epochs], [e[key]*scale for e in epochs], label=str(item['seed']))
    selected = report['selected']
    axes[0, 0].scatter([selected['epoch']], [selected['validation_loss']], marker='*', color='black', s=75, zorder=10, label='验证集选中')
    counts = report['protocol']['split_counts']
    axes[0, 0].set(title=f'LSTM：{counts["train"]}次训练 / {counts["validation"]}次验证', xlabel='训练轮数', ylabel='验证交叉熵')
    axes[0, 1].set(title='LSTM验证准确率（非最终测试成绩）', xlabel='训练轮数', ylabel='准确率 %')
    for ax in axes[0]:
        ax.legend()
        ax.grid(alpha=.2)
    subjects = [g['subject'] for g in report['group_metrics']['subject']]
    xx = np.arange(len(subjects) + 1)
    comparisons = [
        ('LDA', baseline['metrics'], baseline['group_metrics'], '#6a879c'),
        ('CNN', report['cnn_metrics'], report['cnn_group_metrics'], '#208467'),
        ('LSTM', report['metrics'], report['group_metrics'], '#c77f36'),
    ]
    for offset, (label, metrics, groups, color) in zip([-.25, 0, .25], comparisons):
        by_subject = {g['subject']: g['metrics']['accuracy'] for g in groups['subject']}
        values = [metrics['accuracy']] + [by_subject[s] for s in subjects]
        bars = axes[1, 0].bar(xx + offset, np.array(values)*100, width=.24, label=label, color=color)
        axes[1, 0].bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    axes[1, 0].set(xticks=xx, xticklabels=['总体'] + subjects, ylim=(0, 105), ylabel='准确率 %', title=f'同一{report["final_train_trials"]}次训练 / 同一{report["test_trials"]}次测试')
    axes[1, 0].legend(loc='upper center', ncol=3)
    matrix = np.array(report['metrics']['confusion_matrix'])
    axes[1, 1].imshow(matrix, cmap='Oranges')
    classes = report['protocol']['classes']
    axes[1, 1].set(xticks=range(len(classes)), xticklabels=classes, yticks=range(len(classes)), yticklabels=classes, title='LSTM测试混淆矩阵', xlabel='预测', ylabel='真实')
    plt.setp(axes[1, 1].get_xticklabels(), rotation=55, ha='right')
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[1, 1].text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8, color='white' if matrix[i, j] > matrix.max()/2 else '#222222')
    fig.savefig(folder / 'lstm_vs_cnn_vs_lda.png', dpi=150)
    plt.close(fig)
    print(folder / 'lstm_vs_cnn_vs_lda.png')


if __name__ == '__main__':
    main()

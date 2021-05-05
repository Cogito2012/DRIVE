import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    
    labels = ['Recall (%)', 'f-AUC (%)']
    DRIVE = [82.62, 56.20]
    DRIVE_noAtt = [62.73, 53.55]
    DRIVE_invAtt = [36.43, 52.36]

    
    x = np.arange(len(labels))  # the label locations
    width = 0.15
    step = 0.1
    
    fig, ax = plt.subplots(figsize=(5,4))
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    rects1 = ax.bar(x - width, DRIVE, width, label='DRIVE', color='tomato')
    rects1 = ax.bar(x + step / 3, DRIVE_noAtt, width, label='DRIVE (w/o Attention)', color='c')
    rects2 = ax.bar(x + width + 2 * step / 3, DRIVE_invAtt, width, label='DRIVE (inverse Attention)', color='orange')

    ax.set_ylim(0, 101)
    # ax.set_ylabel('Scores', fontsize=fontsize)
    ax.set_yticklabels(range(0, 120, 20), fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc='upper right')

    plt.tight_layout()
    plt.savefig('output/causality_bars.png')
    plt.savefig('output/causality_bars.pdf')

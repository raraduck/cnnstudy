# ch4_dataloader/utils/load_utils.py
import numpy as np
import matplotlib.pyplot as plt

def compare_validations(all_acc_dict):
    trn_acc = [v['train'] for k, v in all_acc_dict.items()]
    val_acc = [v['val'] for k, v in all_acc_dict.items()]
    
    width =0.3
    plt.bar(np.arange(len(trn_acc)), trn_acc, width=width, label='train')
    plt.bar(np.arange(len(val_acc))+ width, val_acc, width=width, label='val')
    plt.xticks(np.arange(len(val_acc))+ width/2, list(all_acc_dict.keys()),
               rotation=60)
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.ylim(0.7, 1)
    plt.savefig('./fig/accuracy_comparison.png', bbox_inches='tight')
    plt.show()


def display_from_batch(sample_batch, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    axes_1d = axes.ravel()
    for r in range(nrows):
        for c in range(ncols):
            i = ncols * r + c
            (imgs, cls) = sample_batch
            axes_1d[i].imshow(imgs[i,:].permute(1,2,0))
            axes_1d[i].set_title(cls[i].item())
    plt.tight_layout()
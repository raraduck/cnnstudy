import os
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

class PlotMaker:
    def __init__(self):
        pass
    
    def draw_plot(self, train_losses, test_losses, title):
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.title(title)
        plt.xlabel("epoch")
        plt.show()
            

def compare_validations(timestamp, all_acc_dict):
    trn_acc = [v['train'] for k, v in all_acc_dict.items()]
    val_acc = [v['val'] for k, v in all_acc_dict.items()]
    
    width =0.3
    plt.bar(np.arange(len(trn_acc)), trn_acc, width=width, label='train')
    plt.bar(np.arange(len(val_acc))+ width, val_acc, width=width, label='val')
    plt.xticks(np.arange(len(val_acc))+ width/2, list(all_acc_dict.keys()), rotation=60)
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.ylim(0.7, 1)
    plt.savefig(f'./_results/{timestamp}_accuracy_comparison.png', bbox_inches='tight')
    plt.show()
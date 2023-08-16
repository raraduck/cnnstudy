import os
import torch
import datetime
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
                 
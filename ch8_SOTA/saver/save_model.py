import os
import torch
import datetime

class ModelSaver:
    def __init__(self, model, rootdir, filename='unknwon', timestamp=None):
        self.model = model
        self.timestamp = timestamp if timestamp else datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        self.timestamp_dir = os.path.join(rootdir, self.timestamp)
        self.filename = filename
        self.best_test_loss = float('inf')
        self.create_folder()
    
    def create_folder(self):
        try:
            if not os.path.exists(self.timestamp_dir):
                os.makedirs(self.timestamp_dir)
                print(f'created new folder: {self.timestamp_dir}')
            else:
                print(f'folder already exist: {self.timestamp_dir}')
        except OSError:
            print(f'creating folder error: {self.timestamp_dir}')
            
    def save_model(self, filename, epoch):
        filename_epoch = filename+'_'+f'{epoch:04}'
        torch.save(self.model.state_dict(), os.path.join(self.timestamp_dir, filename_epoch+'.pt'))
        print(f'saved {filename_epoch}')
        
    def save_at_best_test_loss(self, epoch, test_loss):
        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.save_model(self.filename, epoch)

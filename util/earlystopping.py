import torch
import numpy as np
import os


class EarlyStopping(object):
    def __init__(self, model_path):
        self.count = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.model_path = model_path
        self.criterion = False

    def __call__(self, model_dict, val_loss, step):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model_dict, step)
        elif val_loss > self.best_loss:
            self.count += 1
            print(f'EarlyStopping counter:{self.count}.')
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model_dict, step)
            self.count = 0

    def save_checkpoint(self, val_loss, model_dict, step):
        print(f'Validation loss decreased ({self.val_loss_min:.6f}\
              --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        save_file_name = os.path.join(self.model_path, '%d.pth.tar' % step)
        torch.save(model_dict, save_file_name)
        self.val_loss_min = val_loss

        self.criterion = self.count == self.persistence

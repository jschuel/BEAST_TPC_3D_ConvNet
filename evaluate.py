import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import net
from dataset import Dataset


class evaluate:
    def __init__(self, batch_size, num_workers, PATH, save = True):
        self.data = self.load_data()
        self.PATH = PATH
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net()
        self.net.to(self.device)
        print('Device: %s'%(self.device))
        self.test_ids = self.data.index.to_numpy()
        print('Number of test samples = %s'%(len(self.test_ids)))
        self.testloader = self.define_generators(self.test_ids, batch_size=batch_size, num_workers = num_workers)
        self.pred, self.truth, self.probpos, self.probneg = self.evaluate()
        self.testset = self.data.loc[self.data.index.isin(self.test_ids)]
        self.testset['orig_id'] = self.test_ids
        self.testset.index = [i for i in range(0,len(self.testset))]
        self.testset['prediction'] = np.concatenate(self.pred).ravel()
        self.testset['truth'] = np.concatenate(self.truth).ravel()
        self.testset['probpos'] = np.concatenate(self.probpos).ravel()
        self.testset['probneg'] = np.concatenate(self.probneg).ravel()
        if save == True:
            self.testset.to_feather('data/test_tracks_only_evaluated.feather')
    def load_data(self):
        data = pd.read_feather("data/test_tracks_only.feather")
        return data

    def define_generators(self, test_ids, batch_size, num_workers):        
        test_set = Dataset(test_ids)
        testloader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size)
    
        return testloader
    
    def evaluate(self):
        self.net.load_state_dict(torch.load(self.PATH))
        y_pred_list = []
        y_true_list = []
        y_pred_probpos = []
        y_pred_probneg = []
        with torch.no_grad():
            self.net.eval()
            for data_t, target_t in tqdm(self.testloader):
                data_t, target_t = data_t.to(self.device), target_t.to(self.device)
                outputs_t = self.net(data_t)
                _,pred_t = torch.max(outputs_t, dim=1)
                sm = torch.nn.Softmax(dim = 1)
                y_pred_list.append(pred_t.cpu().numpy())
                y_true_list.append(target_t.cpu().numpy())
                y_pred_probpos.append(sm(outputs_t).cpu().numpy()[:,0])
                y_pred_probneg.append(sm(outputs_t).cpu().numpy()[:,1])
                
        return y_pred_list, y_true_list, y_pred_probpos, y_pred_probneg


if __name__ == '__main__':
    PATH = 'models/trained.pth'
    evaluate(batch_size = 8, num_workers = 12, PATH = PATH)

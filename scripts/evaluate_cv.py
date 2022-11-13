'''Updated 11/12/2022
Script to evaluate a trained 3DCNN using k-folds cross validation. This script will evaluate the positive class output by
evaluating each of the k models used in the k-folds training and will compute the mean of those probabilities for each
event. The number of folds we use for our evaluation, as well as the file path corresponding to the trained model weights
of each fold can be adjusted under the if __name__ =='__main__' clause at the bottom of this script.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append('../ConvNet/')
from model import net
from dataset import Dataset


class evaluate:
    def __init__(self, batch_size, num_workers, PATH, nfolds = 10, save = True):
        self.data = self.load_data()
        self.PATH = PATH #File path of trained model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: %s'%(self.device))
        
        self.test_ids = self.data.index.to_numpy()
        self.testloader = self.define_generators(self.test_ids, batch_size=batch_size, num_workers = num_workers)

        probs = []
        for fold in range(nfolds): #Loop through the number of folds you used when training the model
            print("EVALUATING FOLD %s of %s\n"%(fold+1, nfolds))
            self.net = net()
            self.net.to(self.device)
            self.PATH = PATH+'%s.pth'%(fold+1)
            self.truth, self.prob = self.evaluate() #Evaluate network at each fold 
            probs.append(np.concatenate(self.prob).ravel()) #Record all classification probabilities

        probs = np.array(probs) #make numpy array to evaluate probabilities
        self.data['prob'] = np.array([probs[i] for i in range(0,len(probs))]).T.mean(axis = 1) #compute mean classification probability over all folds
        self.data['prob_err'] = np.array([probs[i] for i in range(0,len(probs))]).T.std(axis = 1) #compute std dev over all folds
        self.data['truth'] = np.concatenate(self.truth).ravel()
        if save == True:
            self.data.to_feather("../data/sample_evaluated.feather") #Save output dataframe
            
    def load_data(self):
        data = pd.read_feather("../data/sample_noHits.feather") #Load data
        return data

    def define_generators(self, test_ids, batch_size, num_workers):
        params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': num_workers}

        # Generators
        test_set = Dataset(test_ids)
        testloader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=params['batch_size'])
    
        return testloader

    def evaluate(self):
        self.net.load_state_dict(torch.load(self.PATH))
        y_true_list = []
        y_pred_prob = []
        with torch.no_grad():
            self.net.eval()
            for data_t, target_t in tqdm(self.testloader):
                data_t, target_t = data_t.to(self.device), target_t.to(self.device)
                outputs_t = self.net(data_t)
                _,pred_t = torch.max(outputs_t, dim=1)
                sm = torch.nn.Softmax(dim = 1) #Compute probabilities using softmax
                y_true_list.append(target_t.cpu().numpy())
                y_pred_prob.append(sm(outputs_t).cpu().numpy()[:,1])
                
        return y_true_list, y_pred_prob


if __name__ == '__main__':
    PATH = '../ConvNet/models/fold'
    nfolds = 10
    evaluate(batch_size = 128, num_workers = 12, PATH = PATH, nfolds = nfolds)

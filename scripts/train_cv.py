'''Updated 11/12/2022
Script to train the 3DCNN using k-folds cross validation with early stopping.
The number of epochs, early stopping rounds, and number of cross validation folds
can be set at the bottom of this file under the if __name__ == '__main__' clause'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append('../ConvNet/')

from model import net
from dataset import Dataset


class train:
    def __init__(self, n_epochs, batch_size, num_workers,  learning_rate, inPATH, outPATH, e_stop = 10, nfold = 10, retrain = False):
        self.data = self.load_data()
        self.inPATH = inPATH    # Use '.' if you don't want to load a pretrained model, otherwise use the file path of the model you'd like to load
        self.nfold = nfold      # Number of cross validation folds
        self.e_stop = e_stop    # Number of successive rounds without validation loss improvement before commencing early stopping
        self.lr = learning_rate
        self.epochs = n_epochs  # Maximum number of epochs if early stoppin isn't initialized
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: %s'%(self.device))

        ###Perform kfold splits###
        self.skf = KFold(n_splits=self.nfold, shuffle = True)
        self.indices = self.data.index.to_numpy() #Indices are fed into the dataloader
        
        for i, (train_index, val_index) in enumerate(self.skf.split(self.indices,self.indices)):
            self.outPATH = outPATH+'%s.pth'%(i+1) #give the fold-number of the model weights in the model name
            self.trainloader, self.valloader = self.define_generators(train_index, val_index, batch_size=batch_size, num_workers = num_workers) #generate data loaders for each split
            self.net = net() #initialize untrained network at each fold
            self.net.to(self.device)
            print('Fold: %s Ntrain = %s\nNval = %s\n'%(i+1,len(train_index),len(val_index)))
            print('PATH of fold: %s'%(self.outPATH))
            print('BEGIN TRAINING FOLD: %s'%(i+1))
            self.train_fold(retrain = retrain) #train the fold
        
    def load_data(self):
        data = pd.read_feather("../data/sample_noHits.feather")
        return data

    def define_generators(self, train_ids, val_ids, batch_size, num_workers):
        params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': num_workers}

        # Generators
        training_set = Dataset(train_ids)
        trainloader = torch.utils.data.DataLoader(training_set, **params)

        validation_set = Dataset(val_ids)
        valloader = torch.utils.data.DataLoader(validation_set, **params)
    
        return trainloader, valloader
    
    def train_fold(self, criterion = nn.CrossEntropyLoss(), retrain = False):
        valid_loss_min = np.inf
        best_e         = 1 #log best epoch for early stopping

        train_loss     = []
        train_acc      = []
        val_loss       = []
        val_acc        = []
        total_step     = len(self.trainloader)
        optimizer      = optim.Adam(self.net.parameters(), lr=self.lr)
        
        if retrain == True:
            self.net.load_state_dict(torch.load(self.inPATH))

        for epoch in range(1, self.epochs+1):
            if epoch - best_e > self.e_stop: #early stopping criteria
                print("Model didn't improve over %s successive epochs. Stop training and move to next fold!\n"%(self.e_stop))
                break
            
            running_loss = 0.0
            correct = 0
            total=0
            print(f'Epoch {epoch}\n')
            for batch_idx, (data_, target_) in enumerate(tqdm(self.trainloader)):
                data_, target_ = data_.to(self.device), target_.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.net(data_).squeeze()
                loss = criterion(outputs, target_)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _,pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred==target_).item()
                total += target_.size(0)

            train_loss.append(running_loss/total_step)
            train_acc.append(100*correct/total)
            print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')

            batch_loss = 0
            total_v    = 0
            correct_v  = 0
            with torch.no_grad():
                self.net.eval()
                for data_v, target_v in (self.valloader):
                    data_v, target_v = data_v.to(self.device), target_v.to(self.device)
                    outputs_v = self.net(data_v).squeeze()
                    
                    loss_v = criterion(outputs_v, target_v)
                    batch_loss += loss_v.item()
                    _,pred_v = torch.max(outputs_v, dim=1)
                    correct_v += torch.sum(pred_v==target_v).item()
                    total_v += target_v.size(0)
                val_loss.append(batch_loss/len(self.valloader))
                val_acc.append(100*correct_v/total_v)
                network_learned = batch_loss < valid_loss_min
                
                print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_v/total_v):.4f}\n')
        
                if network_learned:
                    best_e = epoch
                    valid_loss_min = batch_loss
                    torch.save(self.net.state_dict(), self.outPATH)
                    print('Improvement-Detected, save-model')
            self.net.train()

if __name__ == '__main__':
    max_epoch = 1000                                 #number of training epochs if early stopping doesn't terminate sooner
    inPATH = '.'                                     #Insert a pytorch trained model here and set retrain to True if you'd like to retrain a model
    outPATHbase = '../ConvNet/models/new_train_fold' #Regardless of whether you retrain or train from scratch your model will be saved here
    nfold = 10                                       #Number of cross validation folds
    early_stopping_rounds = 10                       #If the number of successive epochs with no improvement in validation loss exceeds this number, then the training will terminate early.

    ### Train the model ###
    train(n_epochs = max_epoch, batch_size = 128, num_workers = 12, learning_rate = 0.0002, inPATH = inPATH, outPATH = outPATHbase, e_stop = early_stopping_rounds, nfold = nfold, retrain = False)

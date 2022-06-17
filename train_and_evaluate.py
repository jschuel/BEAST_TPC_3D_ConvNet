import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import net
from dataset import Dataset


class train_and_evaluate:
    def __init__(self, n_epochs, batch_size, num_workers,  learning_rate, inPATH, outPATH, save = True, retrain = True):
        self.data = self.load_data()
        self.inPATH = inPATH
        self.outPATH = outPATH
        self.lr = learning_rate
        self.epochs = n_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net()
        self.net.to(self.device)
        print('Device: %s'%(self.device))

        ### Define training and testing split ###
        self.train_ids = self.data[:int(len(self.data)/2)].index.to_numpy()
        self.val_ids = self.data[int(len(self.data)/2):int(len(self.data)/1.6)].index.to_numpy()
        self.test_ids = self.data[int(len(self.data)/1.6):].index.to_numpy()
        
        print('Number training samples = %s\nNumber of validation samples = %s\nNumber of test samples = %s'%(len(self.train_ids),len(self.val_ids),len(self.test_ids)))
        self.trainloader, self.valloader, self.testloader = self.define_generators(self.train_ids, self.val_ids, self.test_ids, batch_size=batch_size, num_workers = num_workers)
        self.train_network(retrain = retrain)
        
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

    def define_generators(self, train_ids, val_ids, test_ids, batch_size, num_workers):
        params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': num_workers}

        # Generators
        training_set = Dataset(train_ids)
        trainloader = torch.utils.data.DataLoader(training_set, **params)

        validation_set = Dataset(val_ids)
        valloader = torch.utils.data.DataLoader(validation_set, **params)

        test_set = Dataset(test_ids)
        testloader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=params['batch_size'])
    
        return trainloader, valloader, testloader
    
    def train_network(self, criterion = nn.CrossEntropyLoss(), retrain = False):
        valid_loss_min = np.Inf
        val_loss = []
        val_acc = []
        train_loss = []
        train_acc = []
        total_step = len(self.trainloader)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        if retrain == True:
            self.net.load_state_dict(torch.load(self.inPATH))
        for epoch in range(1, self.epochs+1):
            running_loss = 0.0
            correct = 0
            total=0
            print(f'Epoch {epoch}\n')
            for batch_idx, (data_, target_) in enumerate(tqdm(self.trainloader)):
                data_, target_ = data_.to(self.device), target_.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.net(data_).squeeze()
                try:
                    loss = criterion(outputs, target_)
                except IndexError:
                    continue
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _,pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred==target_).item()
                total += target_.size(0)
                
            train_acc.append(100 * correct / total)
            train_loss.append(running_loss/total_step)
            print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
            batch_loss = 0
            total_t=0
            correct_t=0
            with torch.no_grad():
                self.net.eval()
                for data_t, target_t in (self.valloader):
                    data_t, target_t = data_t.to(self.device), target_t.to(self.device)
                    outputs_t = self.net(data_t).squeeze()
                    try:
                        loss_t = criterion(outputs_t, target_t)
                    except IndexError:
                        continue
                    batch_loss += loss_t.item()
                    _,pred_t = torch.max(outputs_t, dim=1)
                    correct_t += torch.sum(pred_t==target_t).item()
                    total_t += target_t.size(0)
                val_acc.append(100 * correct_t/total_t)
                val_loss.append(batch_loss/len(self.valloader))
                network_learned = batch_loss < valid_loss_min
                print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(self.net.state_dict(), self.outPATH)
                    print('Improvement-Detected, save-model')
            self.net.train()

    def evaluate(self):
        self.net.load_state_dict(torch.load(self.outPATH))
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
                sm = torch.nn.Softmax(dim = 1) # Use softmax to get output probabilities
                y_pred_probpos.append(sm(outputs_t).cpu().numpy()[:,0])
                y_pred_probneg.append(sm(outputs_t).cpu().numpy()[:,1])
                
        return y_pred_list, y_true_list, y_pred_probpos, y_pred_probneg


if __name__ == '__main__':
    nepoch = 50 #Really we want to use early stopping
    inPATH = '.'
    outPATH = 'models/new_training.pth'
    print(inPATH, outPATH)
    train_and_evaluate(n_epochs = nepoch, batch_size = 8, num_workers = 12, learning_rate = 0.0002, inPATH = inPATH, outPATH = outPATH, save = False, retrain = False)

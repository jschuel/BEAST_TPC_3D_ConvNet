import numpy as np
import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, list_IDs):
        self.list_IDs = list_IDs

  def __len__(self):
        # Gives number of samples
        return len(self.list_IDs)

  def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]
        data = torch.load('tensors/%s.pt'%(ID)) #2-tuple containing [0] a sparse tensor representation of the 3D event charge distribution and [1] the class label
        X = data[0].to_dense() #Convert to dense tensor for training and evaluation
        X = X.unsqueeze(0)
        X = X.float()
        y = data[1] #class label

        return X, y

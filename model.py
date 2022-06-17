import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module): #input images are 34 x 170 x 34
    def __init__(self):
        super(net, self).__init__()
        self.block1 = self.conv_block(c_in=1, c_out=8, dropout=0.1, kernel_size=2, stride=(1,2,1), padding=0) #We downsample aggressively in y
        self.block2 = self.conv_block(c_in=8, c_out=16, dropout=0.1, kernel_size=3, stride=(1,2,1), padding=0)
        self.block3 = self.conv_block(c_in=16, c_out=32, dropout=0.1, kernel_size=(2,3,2), stride=1, padding=0)
        self.block4 = self.conv_block(c_in=32, c_out=64, dropout=0.1, kernel_size=(2,3,2), stride=1, padding=(1,0,1))
        self.block5 = self.conv_block(c_in=64, c_out=32, dropout=0.1, kernel_size=3, stride=1, padding=(1,0,1))
        self.block6 = self.conv_block(c_in=32, c_out=16, dropout=0.1, kernel_size=2, stride=1, padding=0)
        self.block7 = self.conv_block(c_in=16, c_out=8, dropout=0.1, kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(8 * 6 * 6 * 6, 64) 
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)
        self.avgpool2 = nn.AvgPool3d(kernel_size=(2,2,2), stride = 2)
        self.dropout = nn.Dropout(p=0.05) #dropout for FCNN
    def forward(self, x):        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool2(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.avgpool2(x)
        x = self.block7(x)
        x = x.view(-1, 8 * 6 * 6 * 6) #Flatten output featuremap to feed into FCNN
        x = F.selu(self.fc1(x)) #Use SELU activation
        x = self.dropout(x)
        x = F.selu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm3d(num_features=c_out),
            nn.SELU(), #Use SELU activation
            nn.Dropout3d(p=dropout)
        )        
        return seq_block

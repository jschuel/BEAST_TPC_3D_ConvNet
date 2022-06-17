import torch
from model import net
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = net()
model.to(device)
print(model)
summary(model,(1,34,170,34))

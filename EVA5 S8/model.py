from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def network():
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv = nn.Sequential(
              #Convolution Block
              nn.Conv2d(3, 32, 3, padding=1),  #32x32x3 > 32x32x32 : RF: 3x3
              nn.ReLU(),
              nn.BatchNorm2d(32),
              nn.Conv2d(32, 64, 3, padding=1),  #32x32x32 > 32x32x64 : RF: 5x5
              nn.ReLU(),
              nn.BatchNorm2d(64), 
              #transition block
              nn.Conv2d(64, 32, 1),            #32x32x64 > 32x32x32 : RF: 5x5
              nn.MaxPool2d(2,2),              #32x32x64 > 16x16x32 : RF: 6x6
              # Convolution Block
              nn.Conv2d(32, 64, 3, padding= 1),   #16x16x32 > 16x16x64 : RF: 10x10
              nn.ReLU(),
              nn.BatchNorm2d(64), 
              nn.Conv2d(64, 128, 3,  dilation= 2), #16x16x64> 12x12x128 : RF: 18x18
              nn.ReLU(),
              nn.BatchNorm2d(128),
              nn.Conv2d(128, 128, 3, padding= 1), #12x12x128> 12x12x128 : RF: 22x22
              nn.ReLU(),
              nn.BatchNorm2d(128),
              # Transition Block
              nn.Conv2d(128, 64, 1), #12x12x128> 12x12x64 : RF: 22x22
              nn.MaxPool2d(2,2),     #12x12x64> 6x6x64 : RF: 24x24
              
              # Convolution Block
              nn.Conv2d(64, 64, 3, groups= 32, padding= 1), #6x6x64> 6x6x64 : RF: 32x32
              nn.Conv2d(64, 128, 1),  #6x6x64> 6x6x128 : RF: 32x32
              nn.ReLU(),
              nn.BatchNorm2d(128),
              nn.Conv2d(128,128, 3, padding= 1),#6x6x128> 6x6x128 : RF: 40x40
              nn.ReLU(),
              nn.BatchNorm2d(128),
              # Transition Block
              nn.Conv2d(128, 100, 1),#6x6x128> 6x6x100 : RF: 40x40
              nn.ReLU(),
              nn.MaxPool2d(2,2),  #6x6x100> 3x3x100 : RF: 44x44
              #Convolution Block
              nn.Conv2d(100, 128, 3, padding= 1), #3x3x100> 3x3x128 : RF: 60x60
              nn.ReLU(),
              nn.BatchNorm2d(128),
              nn.Conv2d(128, 256, 3, padding= 1),   #3x3x128> 3x3x256 : RF: 76x76
              nn.ReLU(),
              nn.BatchNorm2d(256),
              nn.AvgPool2d(3, 2),       #3x3x256> 1x1x256 : RF: 84x84
              nn.Conv2d(256, 10 , 1)            #1x1x256> 1x1x10 : RF: 84x84
          )

      def forward(self, x):
          x = self.conv(x)
          x = x.view(-1, 10)
          return F.log_softmax(x)


  return Net



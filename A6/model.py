from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.2, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)

def gbn_model():

  dropout_value= 0.0
  class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  #28x28x1 > 28x28x8 : RF: 3x3
            nn.ReLU(),
            GhostBatchNorm(8, 2),
            nn.Dropout(dropout_value),
            nn.Conv2d(8, 12, 3, padding=1), # 28x28x8 > 28x28x12 : RF: 5x5
            nn.ReLU(),
            GhostBatchNorm(12, 2), 
            nn.Dropout(dropout_value),
            #transition block
            nn.Conv2d(12, 6, 1),            # 28x28x12 > 28x28x6  : RF: 5x5
            nn.MaxPool2d(2,2),              # 28x28x6 > 14x14x6   : RF: 6x6
            nn.Conv2d(6, 12, 3),            # 14x14x6  > 12x12x12 : RF: 10x10
            nn.ReLU(),
            GhostBatchNorm(12, 2), 
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3),           # 12x12x12 > 10x10x12 : RF: 14x14
            nn.ReLU(),
            GhostBatchNorm(12, 2),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3, padding= 1),# 10x10x12  > 10x10x12 : RF: 18x18
            nn.ReLU(),
            GhostBatchNorm(12, 2),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2,2),               # 10x10x12 > 5x5x12    : RF: 20x20
            nn.Conv2d(12, 12, 3),            # 5x5x12 > 3x3x12      : RF: 28x28
            nn.ReLU(),
            GhostBatchNorm(12, 2),
            nn.Dropout(dropout_value),
            nn.Conv2d(12, 12, 3, padding= 1),# 3x3x12 > 3x3x12      : RF: 36x36
            nn.ReLU(),
            GhostBatchNorm(12, 2),
            nn.Dropout(dropout_value),
            nn.AvgPool2d(3, 2),              # 3x3x12 > 1x1x12 : RF: 40x40
            nn.Conv2d(12, 10 , 1)            # 1x1x12 > 1x1x10 : RF: 40x40
        )
            
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)

  return Net

def bn_model():

  dropout_value= 0.06

  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv = nn.Sequential(
              nn.Conv2d(1, 8, 3, padding=1),  #28x28x1 > 28x28x8 : RF: 3x3
              nn.ReLU(),
              nn.BatchNorm2d(8),
              nn.Dropout(dropout_value),
              nn.Conv2d(8, 12, 3, padding=1), # 28x28x8 > 28x28x12 : RF: 5x5
              nn.ReLU(),
              nn.BatchNorm2d(12), 
              nn.Dropout(dropout_value),
              #transition block
              nn.Conv2d(12, 6, 1),            # 28x28x12 > 28x28x6  : RF: 5x5
              nn.MaxPool2d(2,2),              # 28x28x6 > 14x14x6   : RF: 6x6
              nn.Conv2d(6, 12, 3),            # 14x14x6  > 12x12x12 : RF: 10x10
              nn.ReLU(),
              nn.BatchNorm2d(12), 
              nn.Dropout(dropout_value),
              nn.Conv2d(12, 12, 3),           # 12x12x12 > 10x10x12 : RF: 14x14
              nn.ReLU(),
              nn.BatchNorm2d(12),
              nn.Dropout(dropout_value),
              nn.Conv2d(12, 12, 3, padding= 1),# 10x10x12  > 10x10x12 : RF: 18x18
              nn.ReLU(),
              nn.BatchNorm2d(12),
              nn.Dropout(dropout_value),
              nn.MaxPool2d(2,2),               # 10x10x12 > 5x5x12    : RF: 20x20
              nn.Conv2d(12, 12, 3),            # 5x5x12 > 3x3x12      : RF: 28x28
              nn.ReLU(),
              nn.BatchNorm2d(12),
              nn.Dropout(dropout_value),
              nn.Conv2d(12, 12, 3, padding= 1),# 3x3x12 > 3x3x12      : RF: 36x36
              nn.ReLU(),
              nn.BatchNorm2d(12),
              nn.Dropout(dropout_value),
              nn.AvgPool2d(3, 2),              # 3x3x12 > 1x1x12 : RF: 40x40
              nn.Conv2d(12, 10 , 1)            # 1x1x12 > 1x1x10 : RF: 40x40
          )

      def forward(self, x):
          x = self.conv(x)
          x = x.view(-1, 10)
          return F.log_softmax(x)

  return Net
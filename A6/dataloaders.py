from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


use_cuda = torch.cuda.is_available()


def data_loaders(batch_size):
  torch.manual_seed(1)

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
      batch_size=batch_size, shuffle=True, **kwargs)


  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=False, transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
      batch_size=batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader
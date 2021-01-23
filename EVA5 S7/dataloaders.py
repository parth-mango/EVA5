from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


use_cuda = torch.cuda.is_available()


def data_loaders(batch_size):
    torch.manual_seed(1)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip()])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True,  **kwargs)
    sampleloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True,  **kwargs)
    testset = datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, **kwargs)
    
    return trainloader, testloader, sampleloader

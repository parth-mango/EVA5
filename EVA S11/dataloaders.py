# ! pip install albumentations==0.4.6
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2



use_cuda = torch.cuda.is_available()

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def data_loaders(batch_size, train_transform, test_transform):
    torch.manual_seed(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    trainset= Cifar10SearchDataset(transform= train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True,  **kwargs)
    sampleloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True,  **kwargs)
    testset= Cifar10SearchDataset(train= False, transform= test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, **kwargs)
    
    return trainloader, testloader, sampleloader

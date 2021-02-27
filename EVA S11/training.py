from __future__ import print_function
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import reg_tech




criterion = nn.CrossEntropyLoss()
lr_list= []



train_losses = []
test_losses = []
train_acc = []
test_acc = []
lr_list= []


def train(model, device, train_loader, optimizer, epoch, scheduler):
  model.train()
  pbar = tqdm(train_loader)
  total= 0
  correct = 0
  processed = 0
  
  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, target)
    train_losses.append(loss)
    loss.backward()
    optimizer.step()
    scheduler.step()
    _, predicted = torch.max(outputs.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()
    lr_list.append(optimizer.param_groups[0]['lr'])
    acc= 100*correct/total
    train_acc.append(acc)
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/total:0.2f}')




from __future__ import print_function
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import reg_tech
l1_reg = reg_tech.l1_reg
train_losses = []
test_losses = []
train_acc = []
test_acc = []

def l1_train(model):
  def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
      l1= 0
      lambda_l1= 0.0005
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      y_pred = model(data)
      loss = F.nll_loss(y_pred, target)
      loss += l1_reg(l1, model, lambda_l1, loss)
      train_losses.append(loss)
      loss.backward()
      optimizer.step()
      pred = y_pred.argmax(dim=1, keepdim=True)  
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)
      pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      train_acc.append(100*correct/processed)

  return train

def l2_train(model):

  def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      y_pred = model(data)
      loss = F.nll_loss(y_pred, target)
      train_losses.append(loss)
      loss.backward()
      optimizer.step()
      pred = y_pred.argmax(dim=1, keepdim=True)  
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)
      pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      train_acc.append(100*correct/processed)

  return train

def l1_l2_train(model):
  def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
      l1= 0
      lambda_l1= 0.0005
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      y_pred = model(data)
      loss = F.nll_loss(y_pred, target)
      loss += l1_reg(l1, model, lambda_l1, loss)
      train_losses.append(loss)
      loss.backward()
      optimizer.step()
      pred = y_pred.argmax(dim=1, keepdim=True)  
      correct += pred.eq(target.view_as(pred)).sum().item()
      processed += len(data)
      pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
      train_acc.append(100*correct/processed)

  return train
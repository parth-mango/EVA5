import torch.optim as optim

def l1_reg(l1, model, lambda_l1, loss_nll):
    for p in model.parameters():
      l1= l1 + p.abs().sum()
      loss= loss_nll + lambda_l1 * l1
    return loss
    
def l1_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return optimizer
    
def l2_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
    return optimizer
	
	
def nil_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return optimizer
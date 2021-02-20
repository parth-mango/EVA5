import torch.optim as optim




    
def hi_optimizer(model, lr_rate):
    optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9)
    return optimizer
    
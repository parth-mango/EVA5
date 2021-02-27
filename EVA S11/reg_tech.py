import torch.optim as optim




    
def hi_optimizer(model, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer
    
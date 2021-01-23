import torch.optim as optim




    
def hi_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer
    
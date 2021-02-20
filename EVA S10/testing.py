from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import reg_tech
import torch.nn.functional as F


train_losses = []
test_losses = []
criterion = nn.CrossEntropyLoss()

test_acc = []
wht_correct_dict= {'img': [],
                  'prediction': [],
                  'target': []}

                  
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            wht_correct = predicted.eq(target.view_as(predicted))
            for index,i in enumerate(wht_correct) :
                i = bool(i)
                if i == False:
                    wht_correct_dict['img'].append(data[index])
                    wht_correct_dict['prediction'].append(predicted[index])
                    wht_correct_dict['target'].append(target[index])
                    
                else:
                    pass
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc_single = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc_single))
    test_acc.append(test_acc_single)
    return test_loss


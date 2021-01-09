from tqdm import tqdm
import torch
import reg_tech
import torch.nn.functional as F
l1_reg = reg_tech.l1_reg

train_losses = []
test_losses = []
train_acc = []
test_acc = []
acc_dict = {'l1+bn': [],
               'l2+bn': [],
               'l1+l2+bn': [],
               'gbn': [],
               'l1+l2+gbn': []}
			   

wht_correct_dict= {'img': [],
                  'prediction': [],
                  'target': []}
               
               
def l1_test(model):
  def test(model, device, test_loader, name):
      model.eval()
      test_loss = 0
      correct = 0
      

      with torch.no_grad():
          for data, target in test_loader:
              l1= 0
              lambda_l1= 0.0005
              data, target = data.to(device), target.to(device)
              output = model(data)
              test_loss = l1_reg(l1, model, lambda_l1, test_loss)
              test_loss += F.nll_loss(output, target, reduction='sum').item()
              pred = output.argmax(dim=1, keepdim=True)
              # wht_correct = pred.eq(target.view_as(pred))
              # for index,i in enumerate(wht_correct) :
              #   i = bool(i)
              #   if i == False:
                  
              #     wht_correct_dict['img'].append(data[index])
              #     wht_correct_dict['prediction'].append(pred[index])
              #     wht_correct_dict['target'].append(target[index])
                  
              #   else:
              #     pass

              correct += pred.eq(target.view_as(pred)).sum().item()
              # print(len(wht_correct_lst))

          

      test_loss /= len(test_loader.dataset)
      test_losses.append(test_loss)
      test_acc_single = 100. * correct / len(test_loader.dataset)
      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          test_acc_single))
      
      test_acc.append(100. * correct / len(test_loader.dataset))      
      acc_dict[name].append(test_acc_single)
      return test_loss
      
  return test

def l2_test(model):
  def test(model, device, test_loader, name):
      model.eval()
      test_loss = 0
      correct = 0

      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              test_loss += F.nll_loss(output, target, reduction='sum').item()
              pred = output.argmax(dim=1, keepdim=True)
              wht_correct = pred.eq(target.view_as(pred))
              for index,i in enumerate(wht_correct) :
                i = bool(i)
                if i == False:
                  
                  wht_correct_dict['img'].append(data[index])
                  wht_correct_dict['prediction'].append(pred[index])
                  wht_correct_dict['target'].append(target[index])
                  
                else:
                  pass

              correct += pred.eq(target.view_as(pred)).sum().item()

      test_loss /= len(test_loader.dataset)
      test_losses.append(test_loss)
      test_acc_single = 100. * correct / len(test_loader.dataset)
      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
         test_acc_single))
      
      test_acc.append(test_acc_single)
      acc_dict[name].append(test_acc_single)
      return test_loss
  return test

def l1_l2_test(model):
  def test(model, device, test_loader, name):
      model.eval()
      test_loss = 0
      correct = 0
      output_list= []
      with torch.no_grad():
          for data, target in test_loader:
              l1= 0
              lambda_l1= 0.0005
              data, target = data.to(device), target.to(device)
              output = model(data)
              test_loss = l1_reg(l1, model, lambda_l1, test_loss)
              test_loss += F.nll_loss(output, target, reduction='sum').item()
              pred = output.argmax(dim=1, keepdim=True)

              correct += pred.eq(target.view_as(pred)).sum().item()

              


      test_loss /= len(test_loader.dataset)
      test_losses.append(test_loss)
      test_acc_single = 100. * correct / len(test_loader.dataset)
      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          test_acc_single))
      
      test_acc.append(test_acc_single)
      acc_dict[name].append(test_acc_single)
      return test_loss
  return test
  
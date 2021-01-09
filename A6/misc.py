
import matplotlib.pyplot as plt
import numpy as np

def misclassified_images(wht_correct_dict_gbn):
    figsize= (20)
    figure = plt.figure(figsize= (20, 15))

    for index in range(1, 25 + 1):
        model_pred = wht_correct_dict_gbn['prediction'][index].item()
        correct_answer = wht_correct_dict_gbn['target'][index].item()
        img= wht_correct_dict_gbn['img'][index]
        img= img.cpu()
        img_arr= np.array(img)
        num_image = plt.subplot(6, 10, index)
        num_image.text(0.5,-0.1, 
                       f'target:{correct_answer} pred:{model_pred}', 
                       size=12, ha="center", 
                       transform=num_image.transAxes,
                       color='w')
        plt.axis('off')
        plt.imshow(img_arr.squeeze(), cmap='gray_r')

def validation_graph(loss_dict):
    fig = plt.figure()
    ax = plt.axes()
    x = range(1,1+len(list(loss_dict.values())[0]))
    for name, loss in loss_dict.items():
        plt.plot(x, loss, label=name)

    plt.title('Validation Loss Curve', fontsize=10, color='w')
    plt.xlabel('Epochs', color='w')
    plt.ylabel('Validation Loss', color='w')
    plt.legend()
    plt.show()


def accuracy_graph(acc_dict):
    fig = plt.figure()
    ax = plt.axes()
    x = range(1,1+len(list(acc_dict.values())[0]))
    for name, acc_single in acc_dict.items():
        plt.plot(x, acc_single, label=name)

    plt.title('Validation Accuracy curve', fontsize=10, color='w')
    plt.xlabel('Epochs', color='w')
    plt.ylabel('Validation Accuracy', color='w')
    plt.legend()
    plt.show()

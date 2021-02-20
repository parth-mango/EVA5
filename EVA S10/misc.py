
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def show_images(trainloader, classes):
    # functions to show an image



    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


def misclassified_images(wht_correct_dict):
    figsize= (20)
    figure = plt.figure(figsize= (20, 15))

    for index in range(1, 25 + 1):
        model_pred = wht_correct_dict['prediction'][index].item()
        correct_answer = wht_correct_dict['target'][index].item()
        img= wht_correct_dict['img'][index]
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

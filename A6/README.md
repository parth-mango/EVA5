# Assignment 6 

The goal of the assignment was to take the best code of the 5th assignment and make it train for 25 epochs with 5 different types of regularization techniques given as:

1. with L1 + BN
2. with L2 + BN
3. with L1 and L2 with BN
4. with GBN
5. with L1 and L2 with GBN

As mentioned in the assignment, we were not supposed to run the code manually for each condition, instead we iterated upon the conditions which turned out to be a successfull design.
Since it allows us to record the performance and  other metrics in the same loop, which marks the completion of next task.
We were also asked to plot graphs for validation loss and validation accuracy for given 5 experiments which can be viewed below.
This task pushed us to understand the importance of collecting data about features and hyperparameters. Before this task, we were looking at validation loss and accuracy as 
as temporary metric. Here we collected both of these metric for the 5 experiments and visualised these with help of graphs.
Looking at the graphs we can easily understand that which models are perfoming better, converging faster and are more stable. 


Another goal was to show 25 misclassified images for the GBN model in to a single image.First we found the images for which the targets(available to us from the labelled dataset)
were not same as predictions(outputs of the model) by converting the output of torch.eq from tensor to boolean. Then we appended the misclassified images, their predictions and 
targets into a dictionary. The output of which was plotted using matplotlib to show misclassified images and their respective predictions and targets.

We also modularized the code by converting different functionalities of code into scripts. We also encountered the issue one faces when the code is distributed in multiple files.


## Observations:
1. The first model i.e. with L1 + BN performed well compared to other models but was inferior in terms of validation loss to GBN model.
2. The implementation of models using L2 gave poor results which might be due to inadequate tuning of parameter 'Lambda'.
3. GBN did perform better than BN owing to split of batched data and calculation of mean and std. deviation thereof.

The misclassified images, Validation loss and Validation accuracy are given below:
 

![alt text](https://raw.githubusercontent.com/curiouswala/EVA/main/A6/misclass.png)

![alt text](https://raw.githubusercontent.com/curiouswala/EVA/main/A6/loss.png)

![alt text](https://raw.githubusercontent.com/curiouswala/EVA/main/A6/acc.png)

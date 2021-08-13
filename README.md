# EVA5-Assignments

The EVA is one of the most exhaustive and updated Deep Vision Program in the world! It is spread over three semester-style phases, each restricted by a qualifying exam.
This repo constitute the assignment of the first phase of the program which revolves around object recognition, detection and segmentation applications using the pytorch framework. 

Let's start with the description of each task, its challenges and solution provided by me in terms of the neural network, techniques used and optimization performed.

### S1 Assignment :

    Task: Elucidate your understaning about basic fundamentals of image namely channels, kernels alongwith Convolution operations, loss backpropagation and how network learns 
    during training.
    
    Solution: A thorough description has beem written in the assignment01.md file about fundamental concepts of DNN and operations like convolution and Backprop.
    
### S2 Assignment :

    Task: Quiz on convolution operation using different kernel sizes, fundamental concepts of receptive fields, calculation of kernel parameters, adkustment of different layers 
    like dense layers and max pooling to make an efficient neural network
    
    Solution: One of the important takeaway from the assigment was a 3x3, convolution followed by Max Pooling has better quality but requires more processing and more
    RAM whereas the 3x3 operation with stride of 2 has poor quality but requires less RAM and less processing. Be
    
### S3 Assignment :

    Task: Quiz on data augmentation strategy like "CutOut" , Numpy and  tensor operations , Gradient in simple mathematical operations, , benefits of 1x1 kernel, non-linearity 
    in neural network.
    
    Solution: Data Augmentation strategy like "Cutout" can improve the accuracy of network as they cause the network to become invariant and learn from other features of image .
    1x1 Convolution requires less computation for reducing the number of channels, uses existing channels to create complex
    channels(instead of re-convolution), requires less number of parameters, it also reduces the burden of channel selection on 3x3. Non linearity like relu or sigmoid make the 
    function differentiable, make loss is non constant and the plays an important role in making neural network " Universal Function Approximator".
    
### S4 Assignment :
    
    Task: Make a network that has: 
          1. 6 Convolution Layers with following kernels(10, 10, 20, 20, 30, 30)
          2. No use of Fully Connected Layers
          3. Use MNIST Dataset
          4. Use Maximum 2 max-pool layers 
          5. The network should have less than 10k Parameters. 
          Achieve >99.4% accuracy under 20 epochs. 
          
    Solution: I started extending the depth of the network by changing the number of channels in small increments like 1 => 8 => 16 => 32 which performed better and achieved the validation
    accuracy of 99.4%.

    I wanted to check how further we can reduce the parameters. I started with around 6k parameters and the accuracy touched 99.3% but didn't go beyond that. Then I used multiple techniques like adding 
    Batch Normalization and learning rate scheduler but finally with 9k parameters i was able to achieve 99.5% accuracy. I also rewrote the network in a sequential manner which 
    is easy to experiment with.
    
    There are 8 convolution layers with Batch Normalization and Relu after every layer except the last layer. Initially I have started with the channel size size of 4 and increased it uniformly till number of channel reach 20. I have an understanding that the memory assigned for 20 channels is equal to next 2**n channels(i.e. 32) but since the solution requires us to work under a parameter restriction so we went ahead with the given archietecture.
    
    
### S5 Assignment :
    Task: The problem has 10 steps with initial model having 6.3 Million Parameters with best test accuracy being 99.24. The tenth step/final model is expected to have:
    a).Use MNIST Dataset
    b). No use of Fully Connected Layers
    c). Use Batch Nomralization, lighter model, Regularization techniques, Global Average Pooling, Image Augmentation Techniques.
    Achieve >99.5% Test accuracy under 10 epochs.
    
    
    Solution: First Step(EVA_A5F1):
              Target:
              Setting up a skeleton
              Setting a basic Architecture with GAP to remove the final layer
              Result: 
              Total Parameters - 19,734
              Best Performance - 98.75%
              Analysis: I have to use batch norm to increase contrast, Performance is way below the goal of 99.4
              
              Second Step(EVA_A5F2):
              Target:
              Using Batchnorm to improve the performance
              Result:
              Total Parameters - 19,942
              Best Performance - 99.12%
              Analysis: There is some over fitting in the model. We should add dropout to improve that. I have to reduce the no of parameters below 10k
              
              Third Step(EVA_A5F3): 
              Target:
              Reduce the no of parameters below 10k
              Add dropout to reduce over fitting
              Adding 1x1 kernels for reducing the no of channles
              Result:
              Total Parameters - 7,216
              Best Performance - 99.32%
              Analysis: I have fixed overfitting using the dropout. The performance is still below the target 99.4%. 
              Upon observing the data we believe introducing rotation transform can improve the performance. I have reduced the number of parameters below 10k
              
              
              Fourth Step(EVA_A5F4): 
              Target:
              Add rotation transform
              Result:
              Total Parameters - 7,216
              Best Performance - 99.42%
              Analysis: Applying random rotation did improve the performance. Using LR scheduler might improve the performance further
              
             
### S6 Assignment :     
    => refer A6 for code
    Task: The goal of the assignment was to take the best code of the 5th assignment and make it train for 25 epochs with 5 different types of regularization techniques given as:

          1. with L1 Regularization + Batch Normalization
          2. with L2 Regularization + Batch Normalization
          3. with L1 Regularization + L2 Regularization + Batch Normalization
          4. with Ghost Batch Normalization
          5. with L1 Regularization + L2 Regularization + Ghost Batch Normalization
          
          Another goal was to show 25 misclassified images for the GBN model in to a single image.
          
          Solution: Following observations were made: 
                    1. The first model i.e. with L1 + BN performed well compared to other models but was inferior in terms of validation loss to GBN model.
                    2. The implementation of models using L2 gave poor results which might be due to inadequate tuning of parameter 'Lambda'.
                    3. GBN did perform better than BN owing to split of batched data and calculation of mean and std. deviation thereof.
          I also modularized the code by converting different functionalities of code into scripts.
          
### S7 Assignment :
    => refer EVA5 S7 for code
    Task: Design a network that:
    a). Use Cifar 10 Dataset
    b). Use Convolution and Transition blocks involving 3 max-pool layers in total.
    c). Total Receptive field must be more than 44
    d). One of the layers must be "Depthwise Separable Convolution"
    e). One of the layers must use "Dialated Convolution"
    f). Use Global Average pooling
    Achieve 80% Accuracy with total Parameters less than 1 Million.
    
    Solution:
    I achieved 83.74% accuracy in 33 Epochs with total number of parameters = 854,862.
    
    FIrst, i used Random Crop and Random Horizontal Flip data augmentation techniques right off the bat to make the network robust to differences in image.
    Depthwise separable convolution layers was followed by 1x1 convolution to maintain output uniformity in dimension. Dialated kernels and maxpool were used
    at appropriate layers tweaking different archietectures. 
    
    Different types of colvolution operation and their specific usage is elucidated in the model. Thorough understanding of convolution operations like:
    1. Dialated Convolution
    2. Transpose Convolution
    3. Depthwise separable Convolution
    4. Grouped Convolution 
    is required to make best use of these operation for an efficient network.
       


              
              
              

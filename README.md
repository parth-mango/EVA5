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

    I wanted to check how further we can reduce the parameters. We started with around 6k parameters and the accuracy touched 99.3% but didn't go beyond that. Then I used multiple techniques like adding 
    Batch Normalization and learning rate scheduler but finally with 9k parameters i was able to achieve 99.5% accuracy. I also rewrote the network in a sequential manner which 
    is easy to experiment with.
    
    There are 8 convolution layers with Batch Normalization and Relu after every layer except the last layer. Initially we have started with the channel size size of 4 and increased it uniformly till number of 
    channel reach 20. We have an understanding that the memory assigned for 20 channels is equal to next 2**n channels(i.e. 32) but since the solution requires us to work under 
    a parameter restriction so we went ahead with the given archietecture.
    
    
          

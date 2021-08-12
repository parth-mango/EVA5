# EVA5-Assignments/S4 Assignment Solution

Given assignment is on MNIST Dataset. <br>
The archietecture requrement involves:<br>
=> 99.4% validation accuracy <br>
=> Less than 20k Parameters <br>
=> Less than 20 Epochs <br>
=> No fully connected layer <br>


## Our Solution: 
First we started expermenting with the provided colab notebook which had cnn network with 6,379,786 number of parameters but the given constraints 
were to make a model with >99.4% validation accuracy and less than 20k parameters.

At first we started with removing bigger layers from the network in order to reduce the parameters but further experimentation showed that a
shallow network even with a large number of parameters doesn't perform as well.

We started extending the depth of the network by changing the number of channels in small increments like 1 => 8 => 16 => 32 which performed better
and achieved the validation accuracy of 99.4%.

We wanted to check how further we can reduce the parameters. We started with around 6k parameters and the accuracy touched 99.3% but didn't go beyond 
that. Then we use multiple techniques like adding Batch Normalization and learning rate scheduler but finally with 9k parameters we were able to
achieve 99.5% accuracy. We also rewrote the network in a sequential manner which is easy to experiment with.

We did around three long sessions during the week to achieve this result in a collaborative manner.

This experience gave us an understanding about how different variable affect the performance of a model. For eg: Importance of depth/no. of channels
vs. Number of parameters; Behaviour of learning rate; Importance of Batch Normalization and its effect on convergence of model.

## Given below is the network archietecture:

            nn.Conv2d(1, 4, 3, padding=1), #28x28x1 > 28x28x4 : RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 4, 3, padding=1), # 28x28x4 > 28x28x4 : RF: 5x5
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1), #28x28x4 >  28x28x8 RF: 7x7
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2,2),
            nn.Conv2d(8, 12, 3, padding=1), #14x14x8  > 14x14x12 : RF: 14x14
            nn.ReLU(),
            nn.BatchNorm2d(12), 
            nn.Conv2d(12, 12, 3, padding= 1), # 14x14x12 > 14x14x12  : RF: 16x16
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, 3, padding= 1), # 14x14x12  > 14x14x16 : RF: 18x18
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 20, 3), # 7x7x16 > 5x5x20 : RF: 36x36
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 10, 3), #5x5x20 > 3x3x10 : RF: 38x38
            nn.AvgPool2d(3, 2), # 3x3x10 > 1x1x10 : RF: 77x77

There are 8 convolution layers with Batch Normalization and Relu after every layer except the last layer. Initially we have started with the channel
size size of 4 and increased it uniformly till number of channel reach 20. We have an understanding that the memory assigned for 20 channels is equal
to next 2**n channels(i.e. 32) but since the solution requires us to work under a parameter restriction so we went ahead with the given archietecture.


## The model summary is as follows:

        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Conv2d-1            [-1, 4, 28, 28]              40
                      ReLU-2            [-1, 4, 28, 28]               0
               BatchNorm2d-3            [-1, 4, 28, 28]               8
                    Conv2d-4            [-1, 4, 28, 28]             148
                      ReLU-5            [-1, 4, 28, 28]               0
               BatchNorm2d-6            [-1, 4, 28, 28]               8
                    Conv2d-7            [-1, 8, 28, 28]             296
                      ReLU-8            [-1, 8, 28, 28]               0
               BatchNorm2d-9            [-1, 8, 28, 28]              16
                MaxPool2d-10            [-1, 8, 14, 14]               0
                   Conv2d-11           [-1, 12, 14, 14]             876
                     ReLU-12           [-1, 12, 14, 14]               0
              BatchNorm2d-13           [-1, 12, 14, 14]              24
                   Conv2d-14           [-1, 12, 14, 14]           1,308
                     ReLU-15           [-1, 12, 14, 14]               0
              BatchNorm2d-16           [-1, 12, 14, 14]              24
                   Conv2d-17           [-1, 16, 12, 12]           1,744
                     ReLU-18           [-1, 16, 12, 12]               0
              BatchNorm2d-19           [-1, 16, 12, 12]              32
                MaxPool2d-20             [-1, 16, 6, 6]               0
                   Conv2d-21             [-1, 20, 6, 6]           2,900
                     ReLU-22             [-1, 20, 6, 6]               0
              BatchNorm2d-23             [-1, 20, 6, 6]              40
                   Conv2d-24             [-1, 10, 4, 4]           1,810
                AvgPool2d-25             [-1, 10, 1, 1]               0
        ================================================================
        Total params: 9,274
        Trainable params: 9,274
        Non-trainable params: 0




## The final performance as illustrated by training logs is:

        loss=0.01653393916785717 batch_id=468: 100%|██████████| 469/469 [00:49<00:00,  9.48it/s]
          0%|          | 0/469 [00:00<?, ?it/s]
        Test set: Average loss: 0.0151, Accuracy: 9951/10000 (99.51%)

        Epoch-15 lr: 0.001
        loss=0.05489780381321907 batch_id=468: 100%|██████████| 469/469 [00:49<00:00,  9.50it/s]
          0%|          | 0/469 [00:00<?, ?it/s]
        Test set: Average loss: 0.0152, Accuracy: 9954/10000 (99.54%)

        Epoch-16 lr: 0.001
        loss=0.018615325912833214 batch_id=468: 100%|██████████| 469/469 [00:50<00:00,  9.36it/s]
          0%|          | 0/469 [00:00<?, ?it/s]
        Test set: Average loss: 0.0153, Accuracy: 9948/10000 (99.48%)

        Epoch-17 lr: 0.001
        loss=0.01525721326470375 batch_id=468: 100%|██████████| 469/469 [00:50<00:00,  9.36it/s]
          0%|          | 0/469 [00:00<?, ?it/s]
        Test set: Average loss: 0.0162, Accuracy: 9949/10000 (99.49%)

        Epoch-18 lr: 0.001
        loss=0.009698312729597092 batch_id=468: 100%|██████████| 469/469 [00:48<00:00,  9.62it/s]

        Test set: Average loss: 0.0154, Accuracy: 9952/10000 (99.52%)

        Epoch-19 lr: 0.0001


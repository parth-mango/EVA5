Q1. What are Channels and Kernels ?

Ans. Channels: Channels represent a collection of similar kind of data. For examlple, when we hear a band playing, there are drums, bass, guitar, and vocals. Each of these components individually represent a channel. On similar grounds, a story about thirsty crow is written using many words. Each word is is made up of letters. There are 26 english alphabets which can be used in different combinations given in the dictionary to represent any idea. Each letter represents a single channel. When we talk about the image data, we usually see a 3 channel coloured image where 3 channels represent red, green, blue pixel values for each spatial point.

Channel is also called feature map. When we detect features in the whole image with help of a single kernel, what we get is a channel. Size of channel is equal to size of whole image.

Kernels: In layman terms, Kernel is nothing but a window that moves over an image to extract features from an image. A kernel is a 3x3 matrix(mostly) with pre-set values which performs an arithmetic operation( usually matrix multiplication/ dot product)on the image at a particular 3x3 space and return a value. A combination of these values when convolved on the whole image by a particular kernel gives us a channel/feature map. A kernel detects simple features like vertical lines/edges, horizontal lines/edges and complex shapes as well. There are different types of kernel namely 1x1, 3x3, 5x5, 7x7, 11x11.


Q2. Why should we (nearly) always use 3x3 kernels?

Ans. There are mainly 3 reasons for using mostly 3x3 kernels:

1. Nvidia Acceleration - Nvidia did a great job of accelerating 3x3 kernel computation for faster perfomance which led to mass adapatation of 3x3 kernels for general purpose feature extraction in state of the art CNN implementation.

2. Less computation Cost- VGG model and its subsequent paper was the first to throw light on the effectivenes of using multiple 3x3 kernels stacked together. For comparision, let's take a 11x11 kernel vs 3x3 kernel. What we realise is that stacking five 3x3 kernels gives us an effective kernel size of 11X11. As we compare the number for parameters, it's 59x = 45x parameters for five 3x3 kernels while 1111x= 121x parameters for 1 11x11 kernel, thereby we can see significant reduction in computational cost for similar feature extraction.

3. Symmetry- Smallest kernel that detect left, right, up and down features. A 3x3 kernel is the smallest kernel which is symmetric from every dimension.


Q3. How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...) ?

Ans. 99 Times

199x199 > 197x197,
197x197 > 195x195,
195x195 > 193x193,
193x193 > 191x191,
191x191 > 189x189,
189x189 > 187x187,
187x187 > 185x185,
185x185 > 183x183,
183x183 > 181x181,
181x181 > 179x179,
179x179 > 177x177,
177x177 > 175x175,
175x175 > 173x173,
173x173 > 171x171,
171x171 > 169x169,
169x169 > 167x167,
167x167 > 165x165,
165x165 > 163x163,
163x163 > 161x161,
161x161 > 159x159,
159x159 > 157x157,
157x157 > 155x155,
155x155 > 153x153,
153x153 > 151x151,
151x151 > 149x149,
149x149 > 147x147,
147x147 > 145x145,
145x145 > 143x143,
143x143 > 141x141,
141x141 > 139x139,
139x139 > 137x137,
137x137 > 135x135,
135x135 > 133x133,
133x133 > 131x131,
131x131 > 129x129,
129x129 > 127x127,
127x127 > 125x125,
125x125 > 123x123,
123x123 > 121x121,
121x121 > 119x119,
119x119 > 117x117,
117x117 > 115x115,
115x115 > 113x113,
113x113 > 111x111,
111x111 > 109x109,
109x109 > 107x107,
107x107 > 105x105,
105x105 > 103x103,
103x103 > 101x101,
101x101 > 99x99,
99x99 > 97x97,
97x97 > 95x95,
95x95 > 93x93,
93x93 > 91x91,
91x91 > 89x89,
89x89 > 87x87,
87x87 > 85x85,
85x85 > 83x83,
83x83 > 81x81,
81x81 > 79x79,
79x79 > 77x77,
77x77 > 75x75,
75x75 > 73x73,
73x73 > 71x71,
71x71 > 69x69,
69x69 > 67x67,
67x67 > 65x65,
65x65 > 63x63,
63x63 > 61x61,
61x61 > 59x59,
59x59 > 57x57,
57x57 > 55x55,
55x55 > 53x53,
53x53 > 51x51,
51x51 > 49x49,
49x49 > 47x47,
47x47 > 45x45,
45x45 > 43x43,
43x43 > 41x41,
41x41 > 39x39,
39x39 > 37x37,
37x37 > 35x35,
35x35 > 33x33,
33x33 > 31x31,
31x31 > 29x29,
29x29 > 27x27,
27x27 > 25x25,
25x25 > 23x23,
23x23 > 21x21,
21x21 > 19x19,
19x19 > 17x17,
17x17 > 15x15,
15x15 > 13x13,
13x13 > 11x11,
11x11 > 9x9,
9x9 > 7x7,
7x7 > 5x5,
5x5 > 3x3,
3x3 > 1x1


Q4. How are kernels initialized? 

Ans. Kernels consist of weights which allow for the detection of particular feature. Kernels are initialized randomly with small random values in most of the use cases. Random initialization of kernels makes convergence faster. In many cases, we use back propagation when training a CNN model to update weights.  If we initialize the kernel with a value say for instance '1' , the convergence takes longer to obtain optimal values for weights. Often  learning gets stuck at local minima which is why it takes comparatively longer to converge. This approach is of great importance with simple neural networks with few hidden layers but doesn't give promising results for deep neural networks.

In some cases, we also use pre learned weights for the kernel to accelerate convergence and achieve optimal weight values which is the core idea of transfer learning.  

Another decent way of initializing kernels is Xavier Initialization where we initialize each weight with a small Gausian value with mean zero and    variance between number of incoming network connections andnumber of outgoing network connections from that layer.

Q5. What happens during the training of a DNN?
Ans. A classic DNN consist of multiple hidden layers. Information first flows from input layer towards the hidden layer, finally error is calculated and it is back propagated back in to the network. This back propagation of error  is needed to update the weights of connections.
The last layers of a DNN are usually the fully connected layers in which every node of a layer is connected to every other node in the next node.

 Before we start to train the network, we decide the number of batches into which we divide the input data so that efficient learning can take place. For a defined number of epochs, the learning of network takes place. The target of the backpropagation is to optimize the weights and decrease the error. The learning may stuck at local minima/local optimum which makes convergence to the global minima/global optimum difficult. 

The learning rate of a network decides how fast or slow a network trains. A low learning rate  takes longer to converge while a higher learning rate can jump over the global minima.

There are two major problems that occur during training of model: 
1. Overfitting 
2. Underfitting

An underfit model is unable the catch the underlying patterns in the data. It is usually because model is simple. To overcome the underfitting, we increase the capacity of model.While, an overfit model learns the data too well that it generalizes poorly. It is because model is very complex. We use dropout/regularization to prevent overfitting. 

All these fundamental processes take place when a deep neural network trains.


## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # 4 convolutional layers - increasing number of filters, decreasing kernel size
        ######### TOY ###################
        self.conv1 = nn.Conv2d(1, 16, 4)    # input 224 x 224 x 1, output 221 x 221 x 16, after 2x2 maxpooling 110 x 110 x 16
        self.conv2 = nn.Conv2d(16, 32, 3)    # input 110 x 110 x 16, output 108 x 108 x 32, after 2x2 maxpooling 54 x 54 x 32
        self.conv3 = nn.Conv2d(32, 64, 2)    # input 54 x 54 x 32, output 53 x 53 x 64, after 2x2 maxpooling 26 x 26 x 64
        self.conv4 = nn.Conv2d(64, 128, 1)    # input 26 x 26 x 64, output 26 x 26 x 128, after 2x2 maxpooling 13 x 13 x 128
        self.dense1 = nn.Linear(13*13*128, 3200)
        self.dense2 = nn.Linear(3200, 512)
        self.dense3 = nn.Linear(512, 136)   # output is 136 (68x2) because we have 68 (x,y) keypoints per image
        ########  TOY ##################
        
        #self.conv1 = nn.Conv2d(1, 32, 4)    # input 224 x 224 x 1, output 221 x 221 x 32, after 2x2 maxpooling 110 x 110 x32
        #self.conv2 = nn.Conv2d(32, 64, 3)   # input 110 x 110 x 32, output 108 x 108 x 64, after 2x2 maxpoooling 54 x 54 x 64
        #self.conv3 = nn.Conv2d(64, 128, 2)  # input 54 x 54 x 64, output 53 x 53 x 128, after 2x2 maxpooling 26 x 26 x 128
        #self.conv4 = nn.Conv2d(128, 256, 1) # input 26 x 26 x 128, output 26 x 26 x 256, after 2x2 maxpooling 13 x 13 x 256
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool that uses a square window of kernel_size=2, stride=2
        self.convPool = nn.MaxPool2d(2, 2)
        
        # conv dropouts - increasing dropouts
        self.convDropout1 = nn.Dropout(0.1)  
        self.convDropout2 = nn.Dropout(0.2)  
        self.convDropout3 = nn.Dropout(0.3)  
        self.convDropout4 = nn.Dropout(0.4)  
        
        # dense dropouts - continue increasing dropouts
        self.denseDropout1 = nn.Dropout(0.5)  
        self.denseDropout2 = nn.Dropout(0.6)  
        
        # input is 43264 (13 x 13 x 256), output is 
        # 10 output channels (for the 10 classes)
        #self.dense1 = nn.Linear(13*13*256, 7200)
        #self.dense2 = nn.Linear(7200, 1000)
        #self.dense3 = nn.Linear(1000, 136)   # output is 136 (68x2) because we have 68 (x,y) keypoints per image
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # we will use F.relu for conv layer activations as suggested by the paper
        # https://arxiv.org/pdf/1710.00977.pdf
        x = self.convDropout1(self.convPool(F.relu(self.conv1(x))))
        x = self.convDropout2(self.convPool(F.relu(self.conv2(x))))
        x = self.convDropout3(self.convPool(F.relu(self.conv3(x))))
        x = self.convDropout4(self.convPool(F.relu(self.conv4(x))))
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # now the dense layers
        x = self.denseDropout1(F.relu(self.dense1(x)))
        x = self.denseDropout2(F.relu(self.dense2(x)))
        x = self.dense3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
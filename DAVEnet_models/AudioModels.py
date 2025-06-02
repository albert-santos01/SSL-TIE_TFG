# Author: David Harwath
import torch
import torch.nn as nn
import torch.nn.functional as F

        
class Davenet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(Davenet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1) # One channel input BW
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40,1), stride=(1,1), padding=(0,0)) # 
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1,11), stride=(1,1), padding=(0,5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1,17), stride=(1,1), padding=(0,8))
        self.pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))

    def forward(self, x):
        if x.dim() == 3: # It has to be bw
            x = x.unsqueeze(1) 

        x = self.batchnorm1(x)      # 1024 x 40 x 1
        x = F.relu(self.conv1(x))   # 1024 x 1 x 128
        x = F.relu(self.conv2(x))   # 1024 x 1 x 256 (kernel 1x11 + padding 0x5)
        x = self.pool(x)            # 512 x 1 x 256
        x = F.relu(self.conv3(x))   # 512 x 1 x 512 (kernel 1x17 + padding 0x8)
        x = self.pool(x)            # 256 x 1 x 512
        x = F.relu(self.conv4(x))   # 256 x 1 x 512 (kernel 1x17 + padding 0x8)
        x = self.pool(x)            # 128 x 1 x 512
        x = F.relu(self.conv5(x))   # 128 x 1 x 1024 (kernel 1x17 + padding 0x8)
        x = self.pool(x)            # 64 x 1 x 1024 It should be 128 This maxpooling shouldn't exist
        x = x.squeeze(2)
        return x
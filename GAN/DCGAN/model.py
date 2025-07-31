import torch 
import torch.nn as nn 

class discriminator(nn.Module):
    def __init__(self,channels_img, features_d):
        super(discriminator,self).__init__()
        

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel
from models.CNNBlocks import DenseBlock
from models.CNNBlocks import ResidualBlock
from models.CNNBlocks import BottleneckBlock
from models.UNet import UNet

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau IFT725UNet.  Un réseau inspiré de UNet
mais comprenant des connexions résiduelles et denses.  Soyez originaux et surtout... amusez-vous!

'''

class IFT725UNet(CNNBaseModel):
    """
     Class that implements a brand new segmentation CNN
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds AlexNet  model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super(IFT725UNet, self).__init__()
        # encoder
        in_channels = 1  # gray image
        self.conv_encoder1 = torch.nn.Sequential(
            UNet._contracting_block(self, in_channels=in_channels, out_channels=64),
            DenseBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.max_pool_encoder1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder2 = torch.nn.Sequential(
            UNet._contracting_block(self, 64, 128),
            DenseBlock(128, 128),
            ResidualBlock(128, 128)
        )
        self.max_pool_encoder2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder3 = torch.nn.Sequential(
            UNet._contracting_block(self, 128, 256),
            BottleneckBlock(256, 64, 256),
            ResidualBlock(256, 256)
        )
        self.max_pool_encoder3 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder4 = torch.nn.Sequential(
            UNet._contracting_block(self, 256, 512),            
            BottleneckBlock(512, 128, 512),
            ResidualBlock(512, 512)
        )
        self.max_pool_encoder4 = nn.MaxPool2d(kernel_size=2)
        # Transitional block
        self.transitional_block = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        # Decode
        self.conv_decoder4 = torch.nn.Sequential(
            UNet._expansive_block(self, 1024, 512, 256),
            BottleneckBlock(256, 64, 256),
            ResidualBlock(256, 256)
        )
        self.conv_decoder3 = torch.nn.Sequential(
            UNet._expansive_block(self, 512, 256, 128),
            DenseBlock(128, 128),
            ResidualBlock(128, 128)
        )
        self.conv_decoder2 = torch.nn.Sequential(            
            UNet._expansive_block(self, 256, 128, 64),
            DenseBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.final_layer = UNet._final_block(self, 128, 64, num_classes)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        # Encode
        encode_block1 = self.conv_encoder1(x)
        encode_pool1 = self.max_pool_encoder1(encode_block1)
        encode_block2 = self.conv_encoder2(encode_pool1)
        encode_pool2 = self.max_pool_encoder2(encode_block2)
        encode_block3 = self.conv_encoder3(encode_pool2)
        encode_pool3 = self.max_pool_encoder3(encode_block3)
        encode_block4 = self.conv_encoder4(encode_pool3)
        encode_pool4 = self.max_pool_encoder4(encode_block4)

        # Transitional block
        middle_block = self.transitional_block(encode_pool4)

        # Decode
        decode_block4 = torch.cat((middle_block, encode_block4), 1)
        cat_layer3 = self.conv_decoder4(decode_block4)
        decode_block3 = torch.cat((cat_layer3, encode_block3), 1)
        cat_layer2 = self.conv_decoder3(decode_block3)
        decode_block2 = torch.cat((cat_layer2, encode_block2), 1)
        cat_layer1 = self.conv_decoder2(decode_block2)
        decode_block1 = torch.cat((cat_layer1, encode_block1), 1)
        final_layer = self.final_layer(decode_block1)
        return final_layer
'''
Fin de votre code.
'''
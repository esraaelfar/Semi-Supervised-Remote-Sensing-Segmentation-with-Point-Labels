import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNetResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(UNetResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        
        self.layer0_conv = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu
        ) 
        self.layer0_pool = self.backbone.maxpool 
        
        self.layer1 = self.backbone.layer1 
        self.layer2 = self.backbone.layer2 
        self.layer3 = self.backbone.layer3 
        self.layer4 = self.backbone.layer4 
        
        self.d4 = DecoderBlock(512, 256, 256)
        
        self.d3 = DecoderBlock(256, 128, 128)
        
        self.d2 = DecoderBlock(128, 64, 64)
        
        self.d1 = DecoderBlock(64, 64, 32)
        
        self.final_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        x0 = self.layer0_conv(x) 
        x0p = self.layer0_pool(x0) 
        
        x1 = self.layer1(x0p) 
        x2 = self.layer2(x1)  
        x3 = self.layer3(x2)  
        x4 = self.layer4(x3)  
        
        x = self.d4(x4, x3)   
        x = self.d3(x, x2)    
        x = self.d2(x, x1)    
        x = self.d1(x, x0)    
        
        x = self.final_up(x)  
        logits = self.final_conv(x) 
        
        return logits

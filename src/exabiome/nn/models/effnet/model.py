import torch
import numpy as np
import pandas as pd
import os
import h5py
import torch.nn as nn

__all__ = ['DepSepBlock', 'InvertedResidualBlock', 'SqueezeExcite',
          'EffNet_b0']

p_list = [
        [[16,96,3,24,3,2,1], [24,144,6,24,3,1,1]], #layer 1
        [[24,144,6,40,5,2,2], [40,240,10,40,5,1,2]], #layer 2 
        [[40,240,10,80,3,2,1],[80,480,20,80,3,1,1], 
                                [80,480,20,80,3,1,1]], #layer 3
        [[80,480,20,112,5,1,2], [112,672,28,112,5,1,2], 
                                 [112,672,28,112,5,1,2]], #layer 4
        [[112,672,28,192,5,2,2],[192,1152,48,192,5,1,2],
         [192,1152,48,192,5,1,2], [192,1152,48,192,5,1,2]], #layer 5
        [[192,1152,48,320,3,2,1]] #layer 6
         ]



class DepSepBlock(nn.Module):
    def __init__(self):
        super(DepSepBlock, self).__init__()
        self.conv1 = nn.Conv1d(32, 32, 3, bias=False, groups=32)
        self.bn1 = nn.BatchNorm1d(32)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(32, 8, 1)
        self.conv3 = nn.Conv1d(8, 32, 1)
        self.conv4 = nn.Conv1d(32, 16, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(16)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bn2(out)
        #out = nn.Identity(out)
        
        return out

    
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_ch:int, mid_ch:int, sq_ch:int,
                 out_ch:int, ks=1, stride: int = 1, padding: int =0):
        super(InvertedResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_ch, mid_ch, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(mid_ch, mid_ch, kernel_size=ks, stride=stride,
                              padding=padding, groups=mid_ch, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_ch)
        self.squeeze = SqueezeExcite(mid_ch, sq_ch)
        self.conv5 = nn.Conv1d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False) 
        self.bn3 = nn.BatchNorm1d(out_ch)
        
        self.downsample = nn.Conv1d(in_ch, out_ch,1, stride=stride)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.squeeze(out)
        out = self.conv5(out)
        out = self.bn3(out)
        
        identity = self.downsample(x)
        out += identity
        out = self.act(out)
        
        return out

    
class SqueezeExcite(nn.Module):
    def __init__(self, ch_in, mid_ch):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(ch_in, mid_ch, kernel_size=1, stride=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(mid_ch, ch_in, kernel_size=1, stride=1)
        )
    def forward(self, x):
        out = self.layer(x)
        return out + x
    
    
class EffNet_b0(nn.Module):
    def __init__(self, param_list=p_list, avg_out=200, out_feats=512, n_classes=18):
        super(EffNet_b0, self).__init__()
        
        self.avg_out = avg_out
        self.out_feats = out_feats
        self.n_classes = n_classes
        self.param_list = param_list

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.silu = nn.SiLU(inplace=True)
        self.layer0 = DepSepBlock()
        self.layer_list = []
        for x in range(len(self.param_list)):
            self.layer_list.append(self._make_layer(self.param_list[x]))
        self.layer_list = nn.Sequential(*self.layer_list)
        self.conv2 = nn.Conv1d(320, self.n_classes, 1, 1)
        self.bn2 = nn.BatchNorm1d(self.n_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=self.avg_out)
        self.fc = nn.Linear(in_features=self.avg_out, 
                            out_features=self.out_feats)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                fan_out = m.kernel_size[0] * m.out_channels
                nn.init.normal_(m.weight, mean=0, std=(np.sqrt(2/fan_out)))
                #.kernel_size[0] * m.conv1.out_channels
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, param_list):
        layers = []
        for x in range(len(param_list)):
            layers.append(InvertedResidualBlock(*param_list[x]))
        return nn.Sequential(*layers)
        
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        
        x = self.layer0(x)
        x = self.layer_list(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x)
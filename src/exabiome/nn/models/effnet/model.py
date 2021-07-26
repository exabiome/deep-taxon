import torch
import numpy as np
import pandas as pd
import os
import h5py
import torch.nn as nn
import pdb
import json

from .. import model, AbstractLit


__all__ = ['DepSepBlock', 'InvertedResidualBlock', 'SqueezeExcite',
          'EffNet_b0',]

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

    
class DepthwiseSeperableBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthwiseSeperableBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, in_ch, 3, bias=False, groups=in_ch)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act = nn.SiLU(inplace=True)
        self.se = SqueezeExcite(in_ch, int(in_ch/4))
        self.conv2 = nn.Conv1d(in_ch, out_ch, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
    
    def forward(self, x):        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.se(out)
        out = self.conv2(out)
        out = self.bn2(out)
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
        #identity = x
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

    
def make_dep_sep(in_ch, out_ch, num_layers):
    ll = [DepthwiseSeperableBlock(in_ch, out_ch)]
    for _ in range(1, num_layers):
        ll.append(DepthwiseSeperableBlock(out_ch, out_ch))
    return nn.Sequential(*ll)   


def make_inv_res(in_ch, mid_ch, sq_ch, out_ch,ks, stride, padding, num_layers):
    ll = [InvertedResidualBlock(in_ch, mid_ch[0], sq_ch[0], 
                                out_ch,ks, stride, padding)]
    for _ in range(1, num_layers):
        ll.append(InvertedResidualBlock(out_ch, mid_ch[1], sq_ch[1], out_ch,
                                        ks, stride, padding))
    return nn.Sequential(*ll)


def make_inv_res_block(inverted_residual_params):
    inv_res = [make_inv_res(*x) for x in inverted_residual_params]
    return nn.Sequential(*inv_res)
      
def make_base(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv1d(1, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.SiLU(inplace=True))

def make_head_layer(head_params, out_classes=20):
    in_ch, out_ch = head_params
    return nn.Sequential(nn.Conv1d(in_ch, 20, kernel_size=1, stride=1),
                         nn.BatchNorm1d(20),
                         nn.SiLU(inplace=True),
                         nn.AdaptiveAvgPool1d(output_size=20),
                         nn.Linear(20, 1))#out_classes))

def get_params():
    f = open('params.json')
    p_list = json.load(f)
    f.close()
    return p_list
    
    
class EffNet(AbstractLit):
    def __init__(self, hparams, model_name):
        super(EffNet, self).__init__(hparams)
        self.param_list = get_params()[model_name]

        self.base = make_base(*self.param_list['base_params'])
        self.dep_sep = make_dep_sep(*self.param_list['dep_sep_params'])
        self.inv_res = make_inv_res_block(self.param_list['inv_res_params'])
        self.head = make_head_layer(self.param_list['head_params'])
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                nn.init.normal_(m.weight, mean=0, std=(np.sqrt(2/fan_out)))
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def _forward_impl(self, x):
        #pdb.set_trace()
        x = x.unsqueeze(1).to(torch.half)
        x = self.base(x)
        x = self.dep_sep(x)
        x = self.inv_res(x)
        x = self.head(x)
        x = x.squeeze()
        return x
    
    def forward(self, x):
        return self._forward_impl(x)

    
@model('effnet-b0')
class EffNetB0(EffNet):
    def __init__(self, hparams):
        super().__init__(hparams, model_name='efficientnet_b0')
        
@model('effnet-b1')
class EffNetB0(EffNet):
    def __init__(self, hparams):
        super().__init__(hparams, model_name='efficientnet_b1')
        
@model('effnet-b2')
class EffNetB0(EffNet):
    def __init__(self, hparams):
        super().__init__(hparams, model_name='efficientnet_b2')
        
@model('effnet-b3')
class EffNetB0(EffNet):
    def __init__(self, hparams):
        super().__init__(hparams, model_name='efficientnet_b3')
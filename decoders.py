import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


##############################            AAPB            ######################################




import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from typing import Optional, Union, Sequence
from typing import Optional, Union, Sequence


class AAPB(nn.Module):
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()

        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)

        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)


        self.conv2 = ConvModule(channels * 2, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = self.avg_pool(x)
        x1 = self.conv1(pooled)

        h_feat = self.h_conv(x1)
        v_feat = self.v_conv(x1)
        
        fused = torch.cat([h_feat, v_feat], dim=1)

        out = self.conv2(fused)
        
        return out
    
    
class DimensionMatchingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DimensionMatchingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        return x
    
    



#######################################  MRPB   ####################################################

class MRPB(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MRPB, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):

        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out

        return outputs


###########################################       MACU      #########################################################


class MACU(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        super(MACU, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)

        assert self.stride in [1, 2]

        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)  #c*2
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.MRPB = MRPB(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1     #c*2
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')
        ###############
        self.caa_factor = AAPB(self.out_channels, 11, 11, None, None)  #True
        ##############
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        MRPB_outs = self.MRPB(pout1)   
        if self.add == True:   
            dout = 0
            for dwout in MRPB_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(MRPB_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))  
        out = self.pconv2(dout)
        ######  caa
        caa=self.caa_factor(x)
        # print(out.shape)
        # print(caa.shape)
        caa=x * caa
        out =caa + out
        #######
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
        
        

def  MACULayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        """
        create a series of multi-scale convolution blocks.
        """
        convs = []
        mscb = MACU(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        convs.append(mscb)
        if n > 1:
            for i in range(1, n):
                mscb = MACU(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
                convs.append(mscb)
        conv = nn.Sequential(*convs)
        return conv


##############################################  Up_BLock   ############################################################################
class Up_BLock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(Up_BLock,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
	        nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
    

########################################  CAG   #######################################################

from .CAG_WT import WTConv2d

import torch.nn.functional as F
class CAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'): 
        super(CAG,self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')
        self.WTConv2d = (WTConv2d(F_l, F_l))
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
                
    def forward(self, g, x):  
        g1 = self.W_g(g)
        x_wt=self.WTConv2d(x)
        x1 = self.W_x(x_wt)
        psi = self.activation(g1+x1)
        psi = self.psi(psi)
        return x*psi
    
###############################################      DDSD_GANet_decoder      ##################################

class DDSD_GANet_d(nn.Module):  
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, cag_ks=3, activation='relu6'):
        super(DDSD_GANet_d,self).__init__()
        Up_BLock_ks = 3 # kernel size for Up_BLock
        self.macu4 = MACULayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
	
        self.Up_BLock3 = Up_BLock(in_channels=channels[0], out_channels=channels[1], kernel_size=Up_BLock_ks, stride=Up_BLock_ks//2)
        self.cag3 = CAG(F_g=channels[1], F_l=channels[1], F_int=channels[1]//2, kernel_size=cag_ks, groups=channels[1]//2)
        self.macu3 = MACULayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.Up_BLock2 = Up_BLock(in_channels=channels[1], out_channels=channels[2], kernel_size=Up_BLock_ks, stride=Up_BLock_ks//2)
        self.cag2 = CAG(F_g=channels[2], F_l=channels[2], F_int=channels[2]//2, kernel_size=cag_ks, groups=channels[2]//2)
        self.macu2 = MACULayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
        self.Up_BLock1 = Up_BLock(in_channels=channels[2], out_channels=channels[3], kernel_size=Up_BLock_ks, stride=Up_BLock_ks//2)
        self.cag1 = CAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=cag_ks, groups=int(channels[3]/2))
        self.macu1 = MACULayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
      
    def forward(self, x, skips):
            

        d4 = self.macu4(x)
        # print(d4.shape)

        d3 = self.Up_BLock3(d4)
                
        # cag3
        x3 = self.cag3(g=d3, x=skips[0])
        

        d3 = d3 + x3
        
        d3 = self.macu3(d3)
        # print(d3.shape)

        d2 = self.Up_BLock2(d3)
        

        x2 = self.cag2(g=d2, x=skips[1])
        

        d2 = d2 + x2 

        d2 = self.macu2(d2)

        d1 = self.Up_BLock1(d2)
        

        x1 = self.cag1(g=d1, x=skips[2])
        

        d1 = d1 + x1 
        
        d1 = self.macu1(d1)
        # print(d1.shape)
        
        return [d4, d3, d2, d1]           
    
    
    



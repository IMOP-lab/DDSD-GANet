import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .decoders import DDSD_GANet_d



class DimensionMatchingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DimensionMatchingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        return x 


    
from .GAN_G import GAN 
from .OCF import OCF

class DDSD_GANet(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, cag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./networks/DDSD_GANet/'):
        super(DDSD_GANet, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = pretrained_dir + '/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = pretrained_dir + '/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = pretrained_dir + '/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = pretrained_dir + '/pvt_v2_b3.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = pretrained_dir + '/pvt_v2_b4.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5() 
            path = pretrained_dir + '/pvt_v2_b5.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()  
            path = pretrained_dir + '/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        
        print('Model %s created, param count: %d' %
                     (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        

        self.decoder = DDSD_GANet_d(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, cag_ks=cag_ks, activation=activation)

        print('Model %s created, param count: %d' %
                     ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
             
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        #########################################################

        self.unet_gan = GAN(3, 3)
        unet_weight_path = '/home/xzh4080/2D框架/networks/DDSD_GANet/400_net_G.pth'
        # unet_weight_path = './400_net_G.pth'
        self.unet_gan.load_state_dict(torch.load(unet_weight_path))
        

        self.conv_gan2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv_gan3 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.conv_gan4 = nn.Conv2d(512, 320, kernel_size=3, stride=2, padding=1)
        self.conv_gan5 = nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1)
        self.ocf2 = OCF([64,64],0)
        self.ocf3 = OCF([128,128],0)
        self.ocf4 = OCF([320,320],0)
        self.ocf5 = OCF([512,512],0)



        #########################################################
        
    def forward(self, x, mode='test'):
        
        
        
        g2,g3,g4,g5 = self.unet_gan(x)
        g2 = self.conv_gan2(g2)
        g3 = self.conv_gan3(g3)
        g4 = self.conv_gan4(g4)
        g5 = self.conv_gan5(g5)

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)                   
        
        x1, x2, x3, x4 = self.backbone(x)    

        
        x1=self.ocf2([x1,g2])
        x2=self.ocf3([x2,g3])
        x3=self.ocf4([x3,g4])
        x4=self.ocf5([x4,g5])

        # decoder
        dec_outs = self.decoder(x4, [x3, x2, x1])   


        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])



        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')



        if mode == 'test':
            return p1#[p4, p3, p2, p1]
        



        return p1#[p4, p3, p2, p1]
    

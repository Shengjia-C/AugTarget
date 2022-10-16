import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .context import CPM, AGCB_Element, AGCB_Patch
from .fusion import *
from .interp import *



__all__ = ['agpcnet']


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.5):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class AGPCNet(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',
                 drop=0.1):
        super(AGPCNet, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(512, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1, drop=drop)

        self.context = CPM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)
        #####################################################################
           
        # self.global_Interp1 = global_Interp(8, 128, 128, 128).cuda()
        self.global_Interp2 = global_Interp(8, 256, 128, 128).cuda()
        self.global_Interp3 = global_Interp(8, 512, 64, 64).cuda()
        # self.fc1 = nn.Linear(4096,2048).cuda()
        # self.fc2 = nn.Linear(4096,2048).cuda()
        # self.fc3 = nn.Linear(2048,2048).cuda()
        #####################################################################

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape   #8, 3, 256, 256
        
        
        
        
        c1, c2, c3, c4 = self.backbone(x) #[8, 128, 128, 128] [8, 256, 64, 64] [8, 512, 32, 32]
        
      
        
        out = self.context(c3)
        
        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True) #torch.Size([8, 512, 64, 64])
        
        out = self.fuse23(out, c2) #8, 256, 64, 64
        
        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True) #8, 256, 128, 128
        
        
         #####################################################################

    
        if out.shape[0] == 8:
            
            out = self.global_Interp2(out) + out
            
            return out,  c4        
            

        else:
            
            out = out
    
        #####################################################################
        
        out = self.fuse12(out, c1)
        
       

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return out


class AGPCNet_Pro(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',
                 drop=0.1):
        super(AGPCNet_Pro, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(512, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1, drop=drop)

        self.context = CPM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape

        c1, c2, c3 = self.backbone(x)

        out = self.context(c3)

        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return out


def agpcnet(backbone, scales, reduce_ratios, gca_type, gca_att, drop):
    return AGPCNet(backbone=backbone, scales=scales, reduce_ratios=reduce_ratios, gca_type=gca_type, gca_att=gca_att, drop=drop)

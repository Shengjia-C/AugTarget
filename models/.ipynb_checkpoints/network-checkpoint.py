import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .context import CPM, AGCB_Element, AGCB_Patch
from .fusion import *





__all__ = ['agpcnet']



# import matplotlib.pylab as plt
# import torchvision
# import os
# import cv2
# def featuremap_visual(feature,
#                       out_dir='/home/chenshengjia/feat_vis/',  # 特征图保存路径文件
#                       save_feature=True,  # 是否以图片形式保存特征图
#                       show_feature=True,  # 是否使用plt显示特征图
#                       feature_title=None,  # 特征图名字，默认以shape作为title
#                       num_ch=16,  # 显示特征图前几个通道，-1 or None 都显示
#                       nrow=4,  # 每行显示多少个特征图通道
#                       padding=10,  # 特征图之间间隔多少像素值
#                       pad_value=1  # 特征图之间的间隔像素
#                       ):
    
#     # feature = feature.detach().cpu()
    
#         b, c, h, w = feature.shape
#         feature = feature[0]
#         feature = feature.unsqueeze(1)

#         if c > num_ch > 0:
#             feature = feature[:num_ch]

#         img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
#         img = img.detach().cpu()
#         img = img.numpy()
#         images = img.transpose((1, 2, 0))
#         images = cv2.resize(images, (1024, 1024))
#         # title = str(images.shape) if feature_title is None else str(feature_title)
#         title = str('global-hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)

#         plt.title(title)
#         plt.imshow(images)
#         if save_feature:
#             # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
#             # plt.savefig(os.path.join(root,'1.jpg'))
#             out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
#             plt.savefig(out_root, dpi=400)

#         if show_feature:        plt.show()



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

        c1, c2, c3= self.backbone(x)
        
        
        # featuremap_visual(c1)
        
        
        out = self.context(c3)

        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)
        
#          #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         import cv2    
#         import numpy as np
    
#         # 1.1 获取feature maps
#         features = out # 尺度大小，如：torch.Size([1,80,45,45])
#         print(features.shape)
#         # 1.2 每个通道对应元素求和
#         heatmap = torch.sum(features, dim=1)  # 尺度大小， 如torch.Size([1,45,45])
#         max_value = torch.max(heatmap)
#         min_value = torch.min(heatmap)
#         heatmap = (heatmap-min_value)/(max_value-min_value)*255

#         heatmap = heatmap.cpu().numpy().astype(np.uint8).transpose(1,2,0)  # 尺寸大小，如：(45, 45, 1)
#         src_size = (125,125)  # 原图尺寸大小
#         heatmap = cv2.resize(heatmap, src_size,interpolation=cv2.INTER_LINEAR)  # 重整图片到原尺寸
#         heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
#         # 保存热力图
#         #cv2.imshow('heatmap',heatmap)
#         cv2.imwrite('/home/chenshengjia/推理结果/heatmap.jpg', heatmap)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


#         #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        
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


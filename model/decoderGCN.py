import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder_GCN(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder_GCN, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        k=3

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.conv_l1 = nn.Conv2d(304, num_classes, kernel_size=(k,1), padding =((k-1)//2,0))
        self.conv_l2 = nn.Conv2d(num_classes, num_classes, kernel_size=(1,k), padding =(0,(k-1)//2))
        self.conv_r1 = nn.Conv2d(304, num_classes, kernel_size=(1,k), padding =((k-1)//2,0))
        self.conv_r2 = nn.Conv2d(num_classes, num_classes, kernel_size=(k,1), padding =(0,(k-1)//2))
        self.br = nn.Sequential(nn.Dropout(0.5),
                                nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
                                BatchNorm(num_classes),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        x_res = self.br(x)
        x = x + x_res

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder_GCN(num_classes, backbone, BatchNorm):
    return Decoder_GCN(num_classes, backbone, BatchNorm)

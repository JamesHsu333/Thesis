import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention.attention import build_PAM_without_filter_beta
from model.GCN.GCN_module import build_GCN
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _SPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_SPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

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

class SPP(nn.Module):
    def __init__(self, backbone, BatchNorm):
        super(SPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 1024

        self.spp1 = _SPPModule(inplanes, 256, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.spp2 = build_GCN(1024, 256, k=15)
        self.spp3 = build_GCN(1024, 256, k=27)
        self.spp4 = build_GCN(1024, 256, k=33)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.bn1 = BatchNorm(1280)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.spp1(x)
        x2 = self.spp2(x)
        x3 = self.spp3(x)
        x4 = self.spp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SPP_with_attention(nn.Module):
    def __init__(self, backbone, BatchNorm):
        super(SPP_with_attention, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 1024

        self.spp1 = _SPPModule(inplanes, 256, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.spp2 = build_PAM_without_filter_beta()
        self.spp2_1 = nn.Conv2d(inplanes, 768, 1, stride=1, bias=False)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.bn1 = BatchNorm(1280)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.spp1(x)
        x2 = self.spp2(x)
        x2 = self.spp2_1(x2)
        x3 = self.global_avg_pool(x)
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_spp(backbone, BatchNorm):
    return SPP(backbone, BatchNorm)

def build_spp_with_attention(backbone, BatchNorm):
    return SPP_with_attention(backbone, BatchNorm)
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import build_backbone
from model.attention.attention import build_PAM, build_CAM
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DANet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 mode='DA', sync_bn=True, freeze_bn=False):
        super(DANet, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
        self.mode=mode

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.head = DANetHead(1024, num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self,x):
        """
            inputs :
                x : input imgs (B X C X H X W)
            returns :
                out[0] : Position attention module (B X C X H X W) + Channel attention module (B X C X H X W)
                out[1] : Position attention module (B X C X H X W)
                out[2] : Channel attention module (B X C X H X W)
        """
        _, _, _, fm = self.backbone(x)

        fms = self.head(fm)
        fms = list(fms)

        if self.mode=='DA':
            fms[0] = F.interpolate(fms[0], x.size()[2:], mode='bilinear', align_corners=True)
            out = fms[0]
        elif self.mode=='PA':
            fms[1] = F.interpolate(fms[1], x.size()[2:], mode='bilinear', align_corners=True)
            out = fms[1]
        else:
            fms[2] = F.interpolate(fms[2],  x.size()[2:], mode='bilinear', align_corners=True)
            out = fms[2]

        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.head]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())

        self.sa = build_PAM(inter_channels)
        self.sc = build_CAM(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self._init_weight()

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
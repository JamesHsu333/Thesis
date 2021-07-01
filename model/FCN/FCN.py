import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import build_backbone
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class FCN(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(FCN, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn

        self.conv = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=15, stride=1, padding=1, bias=False),
                                       BatchNorm(num_classes),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(num_classes),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))

    def forward(self,x):
        _, fm1, fm2, fm3 = self.backbone(x)

        fm3 = self.conv(fm3)


        fm3 = F.interpolate(fm3, fm2.size()[2:], mode='bilinear', align_corners=True)
        fm3 = F.interpolate(fm3, fm1.size()[2:], mode='bilinear', align_corners=True)
        fm3 = F.interpolate(fm3, [257, 257], mode='bilinear', align_corners=True)
        out = F.interpolate(fm3, x.size()[2:], mode='bilinear', align_corners=True)

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
        modules = [self.conv]
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
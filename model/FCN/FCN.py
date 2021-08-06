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

        self.conv = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=1, bias=False),
                                       BatchNorm(num_classes),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))

    def forward(self,x):
        fm1, fm2, fm3, fm4, _ = self.backbone(x)

        out = self.conv(fm4)
        fm4_out = out
        out = F.interpolate(out, fm3.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(out, fm2.size()[2:], mode='bilinear', align_corners=True)
        out= F.interpolate(out, fm1.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=True)

        return out, fm4, fm4_out

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
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.ConvTranspose2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.ConvTranspose2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
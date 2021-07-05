import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import build_backbone
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class UpSampling(nn.Module):
    def __init__(self, in_ch, out_ch, BatchNorm):
        super(UpSampling,self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,2,stride=2,padding=1,output_padding=1),
            BatchNorm(out_ch),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self._init_weight()

    def forward(self, inputs):
        tmp = self.model(inputs)
        return tmp

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        self.conv = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=15, stride=1, padding=(15//2), bias=False),
                                       BatchNorm(num_classes),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(num_classes),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))

        self.up1 = UpSampling(21, 21, BatchNorm)
        self.up2 = UpSampling(21, 21, BatchNorm)
        self.up3 = UpSampling(21, 21, BatchNorm)
        self.up4 = UpSampling(21, 21, BatchNorm)


    def forward(self,x):
        _, _, _, fm3 = self.backbone(x)

        fm3 = self.conv(fm3)
        fm3 = self.up1(fm3)
        fm3 = self.up2(fm3)
        fm3 = self.up3(fm3)
        out = self.up4(fm3)

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
        modules = [self.conv, self.up1, self.up1, self.up2, self.up3, self.up4]
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
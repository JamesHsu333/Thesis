import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import build_backbone
from model.deeplab.aspp import build_aspp
from model.deeplab.spp import build_spp, build_spp_with_attention
from model.deeplab.decoder import build_decoder
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        fm1, fm2, fm3, fm4, _ = self.backbone(input)
        x = self.aspp(fm4)
        x = self.decoder(x)
        aspp_out = x
        x = F.interpolate(x, size=fm3.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=fm2.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=fm1.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x, fm4, aspp_out

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
        modules = [self.aspp, self.decoder]
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

class DeepLab_with_GCN(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_with_GCN, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.spp = build_spp(backbone, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        fm1, fm2, fm3, fm4, _ = self.backbone(input)
        x = self.spp(fm4)
        x = self.decoder(x)
        spp_out = x
        x = F.interpolate(x, size=fm3.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=fm2.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=fm1.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x, fm4, spp_out

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
        modules = [self.spp, self.decoder]
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

class DeepLab_with_attention(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_with_attention, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.spp = build_spp_with_attention(backbone, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        fm1, fm2, fm3, fm4, _ = self.backbone(input)
        x = self.spp(fm4)
        x = self.decoder(x)
        spp_out = x
        x = F.interpolate(x, size=fm3.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=fm2.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=fm1.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x, fm4, spp_out

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
        modules = [self.spp, self.decoder]
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
            for name, m in modules[i].named_parameters():
                if "beta" in name:
                    if isinstance(m, nn.Parameter):
                        if m.requires_grad:
                            yield m

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())



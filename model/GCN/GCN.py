import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import build_backbone
from model.GCN.BR import build_br
from model.GCN.GCN_module import build_GCN
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class GCN(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(GCN, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn

        self.gcn1 = build_GCN(256, num_classes, k=15)
        self.gcn2 = build_GCN(512, num_classes, k=15)
        self.gcn3 = build_GCN(1024, num_classes, k=15)

        self.br1 = build_br(num_classes, BatchNorm)
        self.br2 = build_br(num_classes, BatchNorm)
        self.br3 = build_br(num_classes, BatchNorm)
        self.br4 = build_br(num_classes, BatchNorm)
        self.br5 = build_br(num_classes, BatchNorm)
        self.br6 = build_br(num_classes, BatchNorm)

    def forward(self,x):
        _, fm1, fm2, fm3 = self.backbone(x)

        gc_fm1 = self.br1(self.gcn1(fm1))
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))

        gc_fm3 = F.interpolate(gc_fm3, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = F.interpolate(self.br4(gc_fm2 + gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm1 = F.interpolate(self.br5(gc_fm1 + gc_fm2), [257, 257], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm1), x.size()[2:], mode='bilinear', align_corners=True)

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
        modules = [self.gcn1, self.gcn2, self.gcn3, self.br1, self.br2, self.br3, self.br4, self.br5, self.br6]
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
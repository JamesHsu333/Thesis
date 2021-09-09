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
        
        self.gcn = build_GCN(1024, num_classes, k=15)
        self.br = build_br(num_classes, BatchNorm)

    def forward(self,x):
        fm1, fm2, fm3, fm4, _ = self.backbone(x)

        gc_fm = self.br(self.gcn(fm4))
        gc_fm_out = gc_fm

        gc_fm = F.interpolate(gc_fm, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm = F.interpolate(gc_fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm = F.interpolate(gc_fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(gc_fm, x.size()[2:], mode='bilinear', align_corners=True)

        return out, fm4, gc_fm_out

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
        modules = [self.gcn, self.br]
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

class GCN_Large(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(GCN_Large, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn
        
        self.gcn = build_GCN(1024, num_classes, k=33)
        self.br = build_br(num_classes, BatchNorm)

    def forward(self,x):
        fm1, fm2, fm3, fm4, _ = self.backbone(x)

        gc_fm = self.br(self.gcn(fm4))
        gc_fm_out = gc_fm

        gc_fm = F.interpolate(gc_fm, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm = F.interpolate(gc_fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm = F.interpolate(gc_fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(gc_fm, x.size()[2:], mode='bilinear', align_corners=True)

        return out, fm4, gc_fm_out

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
        modules = [self.gcn, self.br]
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


class GCN_C(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(GCN_C, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn
        
        self.gcn = build_GCN(1024, 1024, k=15)
        self.br = build_br(1024, BatchNorm)
        self.classifier = nn.Sequential(nn.Conv2d(1024, num_classes, 1), 
                                     BatchNorm(num_classes),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))

    def forward(self,x):
        fm1, fm2, fm3, fm4, _ = self.backbone(x)

        gc_fm = self.br(self.gcn(fm4))
        gc_fm = self.classifier(gc_fm)
        gc_fm_out = gc_fm

        gc_fm = F.interpolate(gc_fm, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm = F.interpolate(gc_fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm = F.interpolate(gc_fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(gc_fm, x.size()[2:], mode='bilinear', align_corners=True)

        return out, fm4, gc_fm_out

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
        modules = [self.gcn, self.br, self.classifier]
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

class GCN_Large_C(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(GCN_Large_C, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn
        
        self.gcn = build_GCN(1024, 1024, k=33)
        self.br = build_br(1024, BatchNorm)
        self.classifier = nn.Sequential(nn.Conv2d(1024, num_classes, 1), 
                                     BatchNorm(num_classes),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))

    def forward(self,x):
        fm1, fm2, fm3, fm4, _ = self.backbone(x)

        gc_fm = self.br(self.gcn(fm4))
        gc_fm = self.classifier(gc_fm)
        gc_fm_out = gc_fm

        gc_fm = F.interpolate(gc_fm, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm = F.interpolate(gc_fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm = F.interpolate(gc_fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(gc_fm, x.size()[2:], mode='bilinear', align_corners=True)

        return out, fm4, gc_fm_out

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
        modules = [self.gcn, self.br, self.classifier]
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

class GCN_C_res3(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(GCN_C_res3, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn
        
        self.gcn = build_GCN(512, 512, k=33)
        self.br = build_br(512, BatchNorm)
        self.classifier = nn.Sequential(nn.Conv2d(512, num_classes, 1), 
                                     BatchNorm(num_classes),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))

    def forward(self,x):
        fm1, fm2, fm3, _, _ = self.backbone(x)

        gc_fm = self.br(self.gcn(fm3))
        gc_fm = self.classifier(gc_fm)
        gc_fm_out = gc_fm

        gc_fm = F.interpolate(gc_fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm = F.interpolate(gc_fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(gc_fm, x.size()[2:], mode='bilinear', align_corners=True)

        return out, fm3, gc_fm_out

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
        modules = [self.gcn, self.br, self.classifier]
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

class GCN_C_res2(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(GCN_C_res2, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn
        
        self.gcn = build_GCN(256, 256, k=33)
        self.br = build_br(256, BatchNorm)
        self.classifier = nn.Sequential(nn.Conv2d(256, num_classes, 1), 
                                     BatchNorm(num_classes),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))

    def forward(self,x):
        fm1, fm2, _, _, _ = self.backbone(x)

        gc_fm = self.br(self.gcn(fm2))
        gc_fm = self.classifier(gc_fm)
        gc_fm_out = gc_fm

        gc_fm = F.interpolate(gc_fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(gc_fm, x.size()[2:], mode='bilinear', align_corners=True)

        return out, fm2, gc_fm_out

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
        modules = [self.gcn, self.br, self.classifier]
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

class GCN_C_res1(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(GCN_C_res1, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn
        
        self.gcn = build_GCN(64, 64, k=33)
        self.br = build_br(64, BatchNorm)
        self.classifier = nn.Sequential(nn.Conv2d(64, num_classes, 1), 
                                     BatchNorm(num_classes),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))

    def forward(self,x):
        fm1, _, _, _, _ = self.backbone(x)

        gc_fm = self.br(self.gcn(fm1))
        gc_fm = self.classifier(gc_fm)
        gc_fm_out = gc_fm

        out = F.interpolate(gc_fm, x.size()[2:], mode='bilinear', align_corners=True)

        return out, fm1, gc_fm_out

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
        modules = [self.gcn, self.br, self.classifier]
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

class GCN_C_res(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(GCN_C_res, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.freeze_bn = freeze_bn
        
        self.gcn1 = build_GCN(64, 64, k=33)
        self.gcn2 = build_GCN(256, 256, k=33)
        self.gcn3 = build_GCN(512, 512, k=33)
        self.gcn4 = build_GCN(1024, 1024, k=33)
        self.br1 = build_br(64, BatchNorm)
        self.br2 = build_br(256, BatchNorm)
        self.br3 = build_br(512, BatchNorm)
        self.br4 = build_br(1024, BatchNorm)
        self.classifier = nn.Sequential(nn.Conv2d(1856, num_classes, 1), 
                                     BatchNorm(num_classes),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))

    def forward(self,x):
        fm1, fm2, fm3, fm4, _ = self.backbone(x)

        gc_fm1 = self.br1(self.gcn1(fm1))
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))
        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = torch.cat((gc_fm4, gc_fm3), dim=1)
        gc_fm3 = F.interpolate(gc_fm3, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = torch.cat((gc_fm3, gc_fm2), dim=1)
        gc_fm2 = F.interpolate(gc_fm2, fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm1 = torch.cat((gc_fm2, gc_fm1), dim=1)
        gc_fm = self.classifier(gc_fm1)
        gc_fm_out = gc_fm

        out = F.interpolate(gc_fm, x.size()[2:], mode='bilinear', align_corners=True)

        return out, _, gc_fm_out

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
        modules = [self.gcn1, self.br1, self.gcn2, self.br2, self.gcn3, self.br3, self.gcn4, self.br4, self.classifier]
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
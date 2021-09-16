import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import build_backbone
from model.attention.attention import build_PAM, build_PAM_without_filter, build_PAM_without_filter_alpha, build_PAM_without_filter_beta
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class ANet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(ANet, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.head = ANetHead(1024, num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self,x):
        """
            inputs :
                x : input imgs (B X C X H X W)
            returns :
                out : Position attention module (B X C X H X W)
        """
        fm1, fm2, fm3, fm4, _ = self.backbone(x)
        fm = self.head(fm4)
        attention_out = fm

        fm = F.interpolate(fm, fm3.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, x.size()[2:], mode='bilinear', align_corners=True)
        out = fm

        return out, fm4, attention_out

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
            for name, m in modules[i].named_parameters():
                if "gamma" in name:
                    if isinstance(m, nn.Parameter):
                        if m.requires_grad:
                            yield m

class ANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ANetHead, self).__init__()

        self.sa = build_PAM(in_channels)

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), 
                                   BatchNorm(out_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False))
        self._init_weight()

    def forward(self, x):
        sa_feat = self.sa(x)
        sa_output = self.conv6(sa_feat)

        output = sa_output
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

class ANet_without_filter(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(ANet_without_filter, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.head = ANetHead_without_filter(1024, num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self,x):
        """
            inputs :
                x : input imgs (B X C X H X W)
            returns :
                out : Position attention module (B X C X H X W)
        """
        fm1, fm2, fm3, fm4, _ = self.backbone(x)
        fm = self.head(fm4)
        attention_out = fm

        fm = F.interpolate(fm, fm3.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, x.size()[2:], mode='bilinear', align_corners=True)
        out = fm

        return out, fm4, attention_out

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

class ANetHead_without_filter(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ANetHead_without_filter, self).__init__()

        self.sa = build_PAM_without_filter()

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), 
                                   BatchNorm(out_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False))
        self._init_weight()

    def forward(self, x):
        sa_feat = self.sa(x)
        sa_output = self.conv6(sa_feat)

        output = sa_output
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

class ANet_without_filter_alpha(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(ANet_without_filter_alpha, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.head = ANetHead_without_filter_alpha(1024, num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self,x):
        """
            inputs :
                x : input imgs (B X C X H X W)
            returns :
                out : Position attention module (B X C X H X W)
        """
        fm1, fm2, fm3, fm4, _ = self.backbone(x)
        fm = self.head(fm4)
        attention_out = fm

        fm = F.interpolate(fm, fm3.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, x.size()[2:], mode='bilinear', align_corners=True)
        out = fm

        return out, fm4, attention_out

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
            for name, m in modules[i].named_parameters():
                if "alpha" in name:
                    if isinstance(m, nn.Parameter):
                        if m.requires_grad:
                            yield m

class ANetHead_without_filter_alpha(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ANetHead_without_filter_alpha, self).__init__()

        self.sa = build_PAM_without_filter_alpha()

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), 
                                   BatchNorm(out_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False))
        self._init_weight()

    def forward(self, x):
        sa_feat = self.sa(x)
        sa_output = self.conv6(sa_feat)

        output = sa_output
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

class ANet_without_filter_beta(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(ANet_without_filter_beta, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.head = ANetHead_without_filter_beta(1024, num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self,x):
        """
            inputs :
                x : input imgs (B X C X H X W)
            returns :
                out : Position attention module (B X C X H X W)
        """
        fm1, fm2, fm3, fm4, _ = self.backbone(x)
        fm = self.head(fm4)
        attention_out = fm

        fm = F.interpolate(fm, fm3.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, x.size()[2:], mode='bilinear', align_corners=True)
        out = fm

        return out, fm4, attention_out

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
            for name, m in modules[i].named_parameters():
                if "beta" in name:
                    if isinstance(m, nn.Parameter):
                        if m.requires_grad:
                            yield m


class ANetHead_without_filter_beta(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ANetHead_without_filter_beta, self).__init__()

        self.sa = build_PAM_without_filter_beta()

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), 
                                   BatchNorm(out_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False))
        self._init_weight()

    def forward(self, x):
        sa_feat = self.sa(x)
        sa_output = self.conv6(sa_feat)

        output = sa_output
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

class ANet_best_res3(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(ANet_best_res3, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.head = ANetHead_best_res3(512, num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self,x):
        """
            inputs :
                x : input imgs (B X C X H X W)
            returns :
                out : Position attention module (B X C X H X W)
        """
        fm1, fm2, fm3, _, _ = self.backbone(x)
        fm = self.head(fm3)
        attention_out = fm

        fm = F.interpolate(fm, fm2.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, x.size()[2:], mode='bilinear', align_corners=True)
        out = fm

        return out, fm3, attention_out

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
            for name, m in modules[i].named_parameters():
                if "beta" in name:
                    if isinstance(m, nn.Parameter):
                        if m.requires_grad:
                            yield m


class ANetHead_best_res3(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ANetHead_best_res3, self).__init__()

        self.sa = build_PAM_without_filter_beta()

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), 
                                   BatchNorm(out_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False))
        self._init_weight()

    def forward(self, x):
        sa_feat = self.sa(x)
        sa_output = self.conv6(sa_feat)

        output = sa_output
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

class ANet_best_res2(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(ANet_best_res2, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.head = ANetHead_best_res2(256, num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self,x):
        """
            inputs :
                x : input imgs (B X C X H X W)
            returns :
                out : Position attention module (B X C X H X W)
        """
        fm1, fm2, _, _, _ = self.backbone(x)
        fm = self.head(fm2)
        attention_out = fm

        fm = F.interpolate(fm, fm1.size()[2:], mode='bilinear', align_corners=True)
        fm = F.interpolate(fm, x.size()[2:], mode='bilinear', align_corners=True)
        out = fm

        return out, fm2, attention_out

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
            for name, m in modules[i].named_parameters():
                if "beta" in name:
                    if isinstance(m, nn.Parameter):
                        if m.requires_grad:
                            yield m


class ANetHead_best_res2(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ANetHead_best_res2, self).__init__()

        self.sa = build_PAM_without_filter_beta()

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), 
                                   BatchNorm(out_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False))
        self._init_weight()

    def forward(self, x):
        sa_feat = self.sa(x)
        sa_output = self.conv6(sa_feat)

        output = sa_output
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

class ANet_best_res1(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(ANet_best_res1, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.head = ANetHead_best_res2(64, num_classes, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self,x):
        """
            inputs :
                x : input imgs (B X C X H X W)
            returns :
                out : Position attention module (B X C X H X W)
        """
        fm1, _, _, _, _ = self.backbone(x)
        fm = self.head(fm1)
        attention_out = fm

        fm = F.interpolate(fm, x.size()[2:], mode='bilinear', align_corners=True)
        out = fm

        return out, fm1, attention_out

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
            for name, m in modules[i].named_parameters():
                if "beta" in name:
                    if isinstance(m, nn.Parameter):
                        if m.requires_grad:
                            yield m


class ANetHead_best_res1(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ANetHead_best_res1, self).__init__()

        self.sa = build_PAM_without_filter_beta()

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), 
                                   BatchNorm(out_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False))
        self._init_weight()

    def forward(self, x):
        sa_feat = self.sa(x)
        sa_output = self.conv6(sa_feat)

        output = sa_output
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

class ANet_best_res(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(ANet_best_res, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        #self.attention1 = build_PAM_without_filter_beta()
        #self.attention2 = build_PAM_without_filter_beta()
        self.attention3 = build_PAM_without_filter_beta()
        self.attention4 = build_PAM_without_filter_beta()
        self.classifier = nn.Sequential(nn.Conv2d(1536, num_classes, 1), 
                                   BatchNorm(num_classes),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False))

        self.freeze_bn = freeze_bn

    def forward(self,x):
        """
            inputs :
                x : input imgs (B X C X H X W)
            returns :
                out : Position attention module (B X C X H X W)
        """
        fm1, fm2, fm3, fm4, _ = self.backbone(x)
        #att_fm1 = self.attention1(fm1)
        #att_fm2 = self.attention2(fm2)
        att_fm3 = self.attention3(fm3)
        att_fm4 = self.attention4(fm4)

        att_fm4 = F.interpolate(att_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        att_fm3 = F.interpolate(torch.cat((att_fm4, att_fm3), dim=1), fm2.size()[2:], mode='bilinear', align_corners=True)
        #att_fm2 = F.interpolate(torch.cat((att_fm3, att_fm2), dim=1), fm1.size()[2:], mode='bilinear', align_corners=True)
        att_fm2 = F.interpolate(att_fm3, fm1.size()[2:], mode='bilinear', align_corners=True)
        #att_fm1 = F.interpolate(torch.cat((att_fm2, att_fm1), dim=1), x.size()[2:], mode='bilinear', align_corners=True)
        att_fm1 = F.interpolate(att_fm2, x.size()[2:], mode='bilinear', align_corners=True)
        attention_out = att_fm1
        out = self.classifier(att_fm1)

        return out, _, attention_out

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
        modules = [self.attention1, self.attention2, self.attention3, self.attention4, self.classifier]
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
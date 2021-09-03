import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import build_backbone
from model.attention.attention import build_PAM_without_filter_beta
from model.deeplab.aspp import build_aspp
from model.deeplab.spp import build_spp, build_spp_with_attention
from model.deeplab.decoder import build_decoder
from model.GCN.BR import build_br
from model.GCN.GCN_module import build_GCN
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Net(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(Net, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.attention = build_PAM_without_filter_beta()
        self.attention_classifier = nn.Sequential(nn.Conv2d(1024, num_classes, 1), 
                                     BatchNorm(num_classes),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))

        self.gcn = build_GCN(1024, 1024, k=33)
        self.br = build_br(1024, BatchNorm)
        self.gcn_classifier = nn.Sequential(nn.Conv2d(1024, num_classes, 1), 
                                     BatchNorm(num_classes),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))

        self.decoder = build_decoder(num_classes, BatchNorm)
        self.spp = build_spp(backbone, BatchNorm)

        self.alpha = nn.Parameter(torch.Tensor([1,1,1]))
        self.softmax = nn.Softmax(dim=-1)
        self.freeze_bn = freeze_bn

    def forward(self,x):
        fm1, fm2, fm3, fm4, _ = self.backbone(x)

        gc_fm = self.gcn_classifier(self.br(self.gcn(fm4)))
        spp_fm = self.decoder(self.spp(fm4))
        at_fm = self.attention_classifier(self.attention(fm4))
        
        alpha = self.softmax(self.alpha)
        fm_out = alpha[0]*gc_fm+alpha[1]*spp_fm+alpha[2]*at_fm

        out = F.interpolate(fm_out, fm3.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(out, fm2.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(out, fm1.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=True)

        return out, fm4, fm_out

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
        modules = [self.attention, self.gcn, self.br, self.spp, self.decoder, self.attention_classifier, self.gcn_classifier]
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
                if "alpha" in name or "beta" in name:
                    if isinstance(m, nn.Parameter):
                        if m.requires_grad:
                            yield m
        if isinstance(self.alpha, nn.Parameter):
            if self.alpha.requires_grad:
                yield self.alpha

import torch
import torch.nn as nn
from torchsummary import summary

from model.attention.anet import *
from model.deeplab.deeplab import *
from model.FCN.FCN import *
from model.GCN.GCN import *

model = DeepLab(num_classes=21,
                backbone="resnet",
                output_stride=16,
                sync_bn=None,
                freeze_bn=False).cuda()

model = FCN(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN_C(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN_Large(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = ANet(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = ANet_without_filter(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = ANet_without_filter_alpha(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = ANet_without_filter_beta(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()


summary(model, (3, 513, 513))

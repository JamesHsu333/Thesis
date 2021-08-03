import torch
import torch.nn as nn
from torchsummary import summary

from model.attention.anet import *
from model.deeplab.deeplab import *
from model.FCN.FCN import *
from model.GCN.GCN import *

model3 = DeepLab(num_classes=21,
                backbone="resnet",
                output_stride=16,
                sync_bn=None,
                freeze_bn=False).cuda()

model1 = FCN(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model1 = GCN(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model2 = GCN_C(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model2 = ANet(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = ANet_without_filter(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()


summary(model, (3, 513, 513))

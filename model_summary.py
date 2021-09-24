import torch
import torch.nn as nn
from torchsummary import summary

from model.ablation.net import *
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

model = GCN_Large_C(num_classes=21,
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

model = DeepLab_with_GCN(num_classes=21,
                backbone="resnet",
                output_stride=16,
                sync_bn=None,
                freeze_bn=False).cuda()

model = DeepLab_with_attention(num_classes=21,
                backbone="resnet",
                output_stride=16,
                sync_bn=None,
                freeze_bn=False).cuda()

model = Net(num_classes=21,
                backbone="resnet",
                output_stride=16,
                sync_bn=None,
                freeze_bn=False).cuda()

model = GCN_C_res3(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN_C_res2(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN_C_res1(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN_C_res(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = Net_Cat(num_classes=21,
                backbone="resnet",
                output_stride=16,
                sync_bn=None,
                freeze_bn=False).cuda()

model = ANet_best_res3(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = ANet_best_res2(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = ANet_best_res1(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = ANet_best_res(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN_3(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN_2(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN_1(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = GCN_res(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model = ANet_without_filter_gamma(num_classes=21,
            backbone="resnet",
            output_stride=16,
            gamma=1,
            sync_bn=None,
            freeze_bn=False).cuda()

summary(model, (3, 513, 513))

import torch
import torch.nn as nn
from torchsummary import summary

from model.deeplab.deeplab import *
from model.FCN.FCN import *
from model.GCN.GCN import *

model1 = DeepLab(num_classes=21,
                backbone="resnet",
                output_stride=16,
                sync_bn=None,
                freeze_bn=False).cuda()

model = FCN(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()

model2 = GCN(num_classes=21,
            backbone="resnet",
            output_stride=16,
            sync_bn=None,
            freeze_bn=False).cuda()


summary(model, (3, 513, 513))

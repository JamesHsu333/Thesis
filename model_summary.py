from model.deeplab import *

import torch
import torch.nn as nn
from torchsummary import summary
"""
model = DeepLab(num_classes=21,
                backbone="resnet",
                output_stride=16,
                sync_bn=None,
                freeze_bn=False).cuda()
"""
model = DeepLab_GCN(num_classes=21,
                backbone="resnet",
                output_stride=16,
                sync_bn=None,
                freeze_bn=False).cuda()
summary(model, (3, 513, 513))

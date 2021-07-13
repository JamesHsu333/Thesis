import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(nn.Conv2d(1280, num_classes, 1), 
                                     BatchNorm(num_classes),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))
        self._init_weight()


    def forward(self, x):
        x = self.decoder(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, BatchNorm):
    return Decoder(num_classes, BatchNorm)

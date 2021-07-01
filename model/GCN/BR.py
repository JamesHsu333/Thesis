import torch
import torch.nn as nn
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class BR(nn.Module):
    def __init__(self, out_c, BatchNorm):
        super(BR, self).__init__()
        self.bn = BatchNorm(out_c)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
        self._init_weight()
    
    def forward(self,x):
        x_res = x
        x_res = self.conv1(x_res)
        x_res = self.bn(x_res)
        x_res = self.relu(x_res)
        x_res= self.drop1(x_res)
        x_res = self.conv2(x_res)
        x_res= self.drop2(x_res)
        
        x = x + x_res
        
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

def build_br(num_classes, BatchNorm):
    return BR(num_classes, BatchNorm)
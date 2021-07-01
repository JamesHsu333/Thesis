import torch
import torch.nn as nn

class GCN_module(nn.Module):
    def __init__(self,c,out_c,k=7):
        super(GCN_module, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k,1), padding =((k-1)//2,0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1,k), padding =(0,(k-1)//2))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k), padding =((k-1)//2,0))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k,1), padding =(0,(k-1)//2))
        self._init_weight()

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        
        x = x_l + x_r
        
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

def build_GCN(c, num_classes, k):
    return GCN_module(c, num_classes, k)
import torch
import torch.nn as nn


class BRU(nn.Module):
    """
    Bottleneck Residual Unit
    """
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super(BRU, self).__init__()
        bt_ch = in_ch // 4
        self.process = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, bt_ch, kernel_size=1, bias=False),

            nn.BatchNorm2d(bt_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bt_ch, bt_ch, kernel_size=3, stride=stride, 
                      padding=dilation, dilation=dilation, bias=False),

            nn.BatchNorm2d(bt_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bt_ch, out_ch, kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        return self.process(x)


class UR_Block(nn.Module): 
    """
    Up-sampling & Refinement block (UR-block)
    """
    def __init__(self, in_ch, out_ch, k=2):
        super(UR_Block, self).__init__()
        self.k = k
        self.M = BRU(in_ch, out_ch)
        self.upsample = nn.Upsample(scale_factor=2)

    def r(self, x):
        batch, channel, height, width = x.size()
        x = x.view(batch, channel//self.k, self.k, height, width).sum(2)
        
        return x   

    def forward(self, x):
        out = self.r(x) + self.M(x)
        out = self.upsample(out)

        return out


class DR_Block(nn.Module):
    """
    Down-sampling & Refinement block (DR-block)
    """
    def __init__(self, in_ch, out_ch):
        super(DR_Block, self).__init__()
        self.M = BRU(in_ch, out_ch)
        self.downsample = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        out = x + self.M(x)
        out = self.downsample(out)

        return out
    

class SE_Block(nn.Module):
    """
    Squeeze and Excitation block
    """
    def __init__(self, in_ch, out_ch):
        super(SE_Block, self).__init__()
        self.process = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch // 16, kernel_size=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 16, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.process(x) * x + x
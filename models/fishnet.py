import math
import torch
import torch.nn as nn
from models.block import SE_Block
from models.fish import Fish


class FishNet(nn.Module):
    """
    FishNet 구성
    """
    def __init__(self, n_stage:int, in_ch:int, n_class:int, dropout:float):
        super(FishNet, self).__init__()
        self.n_stage = n_stage
        self.fish = Fish(n_stage, in_ch)
        
        self.first_block = nn.Sequential(
            self.conv_bn_relu(3, in_ch//2),
            self.conv_bn_relu(in_ch//2, in_ch//2),
            self.conv_bn_relu(in_ch//2, in_ch),
            nn.MaxPool2d(2, stride=2),
            )        
        self.fish_tail= self.fish.tail()
        tail_last_ch = self.fish.tail_chs[-1]

        self.bridge = SE_Block(in_ch=tail_last_ch, out_ch=tail_last_ch)
        
        self.fish_body = self.fish.body()
        self.fish_head = self.fish.head()
        head_last_ch = self.fish.head_chs[-1]

        self.last_block = nn.Sequential(
            nn.BatchNorm2d(head_last_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_last_ch, head_last_ch//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_last_ch//2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(head_last_ch//2, n_class, kernel_size=1, bias=True),           
            )  
        self.fc = nn.Sequential(nn.Dropout(dropout),
            nn.Linear(head_last_ch//2, head_last_ch//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_last_ch//2, n_class),
            )

    def conv_bn_relu(self, in_ch, out_ch, stride=1):
        """
        first_block을 구성하는 layer
        (Convolution -> Batch Normalization -> ReLU)
        """
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, 
                                       stride=stride, bias=False),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU(inplace=True))
    
    def forward(self, x):
        # First Block
        x = self.first_block(x)

        # Fish Tail
        feats_tail = [x]
        for i in range(self.n_stage):
            x = self.fish_tail[i](x)
            feats_tail.append(x)

        # Bridge
        x = self.bridge(x)
        
        # Fish Body
        feats_body = []
        for i in range(self.n_stage):
            feats_body.append(x)
            cat_x = torch.cat([x, feats_tail[-1-i]], dim=1) 
            x = self.fish_body[i](cat_x)
        
        # Fish Head
        feats_body.append(feats_tail[0])
        for i in range(self.n_stage):
            cat_x = torch.cat([x, feats_body[-1-i]], dim=1) 
            x = self.fish_head[i](cat_x)

        # Last Block
        out = self.last_block(x) 
        out = out.view(x.size(0), -1) 
        # FC
        out = self.fc(out)
        
        return out
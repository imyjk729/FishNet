import torch.nn as nn
from models.block import BRU, UR_Block, DR_Block


class Fish():
    def __init__(self, n_stage:int, in_ch:int, k=2):
        self.n_stage = n_stage
        self.in_ch = in_ch
        self.k = k
        self.tail_chs = [in_ch]
        self.body_chs = []
        self.head_chs = []

    def tail(self):
        """
        fish tail architecture 구성
        channel : 64 -> 128 -> 256 -> 512
        size : 16X16 -> 8X8 -> 4X4 -> 2X2
        """
        tail_block = nn.ModuleList()

        for i in range(self.n_stage):
            self.tail_chs.append(self.tail_chs[i]*2)
            layer = nn.Sequential(
                BRU(self.tail_chs[i], self.tail_chs[i+1]),
                nn.MaxPool2d(self.k, stride=self.k)
            )
            tail_block.append(layer)

        return tail_block

    def body(self):
        """
        fish body architecture 구성
        channel : 512 -> 512 -> 384 -> 256
        size : 2X2 -> 4X4 -> 8X8 -> 16X16
        """
        body_block = nn.ModuleList()
        self.body_chs.append(self.tail_chs[-1])

        for i in range(self.n_stage):
            self.body_chs.append((self.body_chs[i] + self.tail_chs[-1-i])//self.k)
            layer = nn.Sequential(
                UR_Block(self.body_chs[i+1] * self.k, self.body_chs[i+1], self.k),
            )
            body_block.append(layer)

        return body_block
    
    
    def head(self):
        """
        fish head architecture 구성
        channel : 256 -> 320 -> 704 -> 1216
        size : 16X16 -> 8X8 -> 4X4 -> 2X2
        """
        head_block = nn.ModuleList()
        self.head_chs.append(self.body_chs[-1])
        self.body_chs[-1] = self.tail_chs[0]
        
        for i in range(self.n_stage):
            self.head_chs.append(self.body_chs[-1-i]+self.head_chs[i])
            layer = nn.Sequential(
                DR_Block(in_ch=self.head_chs[i+1], out_ch=self.head_chs[i+1]),
            )
            head_block.append(layer)

        return head_block
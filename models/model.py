
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def weight_init(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ResBlock(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5, use_batch_norm=True):
        super(ResBlock, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.use_batch_norm = use_batch_norm

        self.w1 = nn.Linear(self.l_size, self.l_size)
        if use_batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        if use_batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        residual = x

        y = self.w1(x)
        if self.use_batch_norm:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        if self.use_batch_norm:
            y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = residual + y
        return out

class Lifter(nn.Module):

    def __init__(self, linear_size=1024,
                        num_stage=4,
                        p_dropout=0.5):
        super(Lifter, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size = 17*2
        # 3d joints
        self.output_size = 17

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(ResBlock(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout) 

        for m in self.modules():
            weight_init(m)

    def forward(self, x):
        """
        x: (BS, 17*2)
        y: (BS, 17)
        """
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        return y
    

class Discriminator(nn.Module):
    """
    determine whether 2d pose is real or fake. 
    """
    
    def __init__(self, linear_size=1024,
                        num_stage=3,
                        p_dropout=0.5
                        ):

        super(Discriminator, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        self.input_size =  17*2
        self.output_size = 1

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(ResBlock(self.linear_size, self.p_dropout, use_batch_norm=False))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)


        for m in self.modules():
            weight_init(m)
    

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        y = F.sigmoid(y)
        return y


class Discriminator_wgan(nn.Module):
    """
    determine whether 2d pose is real or fake. 
    """
    
    def __init__(self, linear_size=1024,
                        num_stage=3,
                        p_dropout=0.5
                        ):

        super(Discriminator_wgan, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        self.input_size =  17*2
        self.output_size = 1

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(ResBlock(self.linear_size, self.p_dropout, use_batch_norm=False))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        for m in self.modules():
            weight_init(m)
    

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        return y
import torch
import torch.nn as nn
from model.invomlp import InvoMLP_B

class SSIN(nn.Module):
    def __init__(self, band, classes, Dataset, ikernel, group=16):
        super(SSIN, self).__init__()
        dim = 96
        self.conv11 = nn.Conv2d(band, dim, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = InvoMLP_B(dim, ikernel, group)
        self.block2 = InvoMLP_B(dim, ikernel, group)
        self.block3 = InvoMLP_B(dim, ikernel, group)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(dim, classes)

        if Dataset == 'UP':
            self.num = 4
        else:
            self.num = 6

        self.name = 'SSIN_dim' + str(dim) + '_num_' + str(self.num)
        self.fac = dim // self.num
        self.d = dim
        self.fc1 = nn.Linear(2, self.num)
        self.fc2 = nn.Linear(self.num, self.num)
        self.fc3 = nn.Linear(self.num, self.num)
    def forward(self, x, c):
        x = x.squeeze(1).permute(0,3,1,2)
        b,_,h,w = x.size()
        c1 = self.fc1(c)  # 2 - 6
        c2 = self.fc2(c1)  # 6 - 6
        c3 = self.fc3(c2)  # 6 - 6

        x = self.bn1(self.relu(self.conv11(x)))

        x = x+((c1).repeat(1,self.fac).unsqueeze(-1).unsqueeze(-1))

        x1 = self.block1(x)
        x1 = x1+((c2).repeat(1,self.fac).unsqueeze(-1).unsqueeze(-1))

        x2 = self.block2(x1)
        x2 = x2+((c3).repeat(1,self.fac).unsqueeze(-1).unsqueeze(-1))
        x3 = self.block3(x2)

        x = self.avg(x3).squeeze(-1).squeeze(-1)
        x = self.cls(x)

        return x




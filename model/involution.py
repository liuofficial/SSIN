import torch
from torch import nn, select

class Involution(nn.Module):
    def __init__(self, kernel_size, in_channel=4, stride=1, group=1, ratio=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.stride = stride
        self.group = group
        assert self.in_channel % group == 0
        self.group_channel = self.in_channel // group
        self.conv1 = nn.Conv2d(
            self.in_channel,
            self.in_channel // ratio,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(in_channel // ratio)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            self.in_channel // ratio,
            self.group * self.kernel_size * self.kernel_size,
            kernel_size=1
        )
        self.avgpool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2))
        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2))

    def forward(self, inputs):
        B, C, H, W = inputs.shape

        weight = self.conv2(self.relu(self.bn(self.conv1(inputs))))  # (bs,G*K*K,H//stride,W//stride)
        b, c, h, w = weight.shape
        weight = weight.reshape(b, self.group, self.kernel_size * self.kernel_size, h, w).unsqueeze(
            2)  # (bs,G,1,K*K,H//stride,W//stride)

        x_unfold = self.unfold(inputs)

        x_unfold = x_unfold.reshape(B, self.group, C // self.group, self.kernel_size * self.kernel_size,
                                    H // self.stride, W // self.stride)  # (bs,G,C//G,K*K,H//stride,W//stride)

        out = (x_unfold * weight).sum(dim=3)  # (bs,G,G//C,1,H//stride,W//stride)

        out = out.reshape(B, C, H // self.stride, W // self.stride)  # (bs,C,H//stride,W//stride)

        return out


import math
import os
from turtle import forward
from typing import List

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tcm
import pytorch_msssim
import mstr
import utils


class BasicBlock(nn.Module):
    def __init__(self, ch=64, ks_crf=7, ks_ncrf=21):
        super(BasicBlock, self).__init__()
        self.k1 = ks_crf
        self.k2 = ks_ncrf

        self.CRF = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ks_crf, padding=ks_crf // 2, groups=ch, bias=False),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )

        self.nCRF = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ks_ncrf, padding=ks_ncrf // 2, groups=ch,
                      bias=False),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )

        self.combine = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1)

    def forward(self, x):
        x = self.CRF(x)
        x = self.nCRF(x)
        x = self.combine(x)
        return x


class BasicBlock2(nn.Module):
    def __init__(self, ch=64):
        super(BasicBlock2, self).__init__()
        self.or1 = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=5, padding=2, bias=False, groups=ch),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )
        self.or2 = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=5, padding=2, bias=False, groups=ch),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )
        self.or3 = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=5, padding=2, bias=False, groups=ch),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )
        self.or4 = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=5, padding=2, bias=False, groups=ch),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )
        self.combine = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1)

    def forward(self, x):
        o1 = self.or1(x)
        o2 = self.or2(x)
        o3 = self.or3(x)
        o4 = self.or4(x)
        res = self.combine(o1 + o2 + o3 + o4)
        return res


class XChannel(nn.Module):
    def __init__(self, ch=64, ks_crf=7, ks_ncrf=21):
        super(XChannel, self).__init__()
        self.k1 = ks_crf
        self.k2 = ks_ncrf

        self.CRF = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ks_crf, padding=ks_crf // 2, groups=ch, bias=False),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )

        self.inh_w = nn.Parameter(torch.Tensor([0.]))
        self.nCRF = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ks_ncrf, padding=ks_ncrf // 2, groups=ch,
                      bias=False),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )

        self.combine = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1)

    def regist_mask(self):
        r = self.k2 // 2
        cx, cy = torch.meshgrid(torch.linspace(start=-r, end=r, steps=self.k2),
                                torch.linspace(start=-r, end=r, steps=self.k2))
        dist = torch.sqrt(cx * cx + cy * cy)
        idx = (dist < r) & (dist > (self.k1 // 2) * 1.414)
        idx = einops.repeat(idx, 's1 s2 -> n c s1 s2', n=self.nCRF[0].weight.shape[0], c=1)
        self.register_buffer(name='mask', tensor=idx, persistent=False)

    def set_kernnel(self):
        r = self.k2 // 2
        cx, cy = torch.meshgrid(torch.linspace(start=-r, end=r, steps=self.k2),
                                torch.linspace(start=-r, end=r, steps=self.k2))
        dist = torch.sqrt(cx * cx + cy * cy)
        idx = (dist < r) & (dist > (self.k1 // 2) * 1.414)
        idx = einops.repeat(idx, 's1 s2 -> n c s1 s2', n=self.nCRF[0].weight.shape[0], c=1)
        self.nCRF[0].weight.data[~idx] = 0.0
        print('Set X nCRF.')

    def forward(self, x):
        r1 = self.CRF(x)
        if self.training:
            self.nCRF[0].weight.data[~self.mask] = 0.0
        r2 = self.nCRF(x)
        r = self.combine(r1 - self.inh_w * r2)
        return r


class NLU(nn.Module):
    def __init__(self, ncrf_size, ch=64, k=3):
        super(NLU, self).__init__()
        self.range = ncrf_size
        self.k = k
        self.subunit = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, groups=ch, bias=False),
                    nn.InstanceNorm2d(num_features=ch),
                    nn.ReLU(inplace=True)
                ) for i in range(k)
            ]
        )
        self.collect = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ncrf_size, padding=ncrf_size // 2, groups=ch,
                              bias=False),
                    # nn.InstanceNorm2d(num_features=ch)
                ) for i in range(k)
            ]
        )

    def regist_mask(self):
        r = self.range // 2
        width = r // self.k
        cx, cy = torch.meshgrid(torch.linspace(start=-r, end=r, steps=self.range),
                                torch.linspace(start=-r, end=r, steps=self.range))
        dist = torch.sqrt(cx * cx + cy * cy)
        for i, m in enumerate(self.collect):
            idx = (dist < (r - i * width)) & (dist > (r - i * width - width))
            idx = einops.repeat(idx, 's1 s2 -> n c s1 s2', n=m[0].weight.shape[0], c=1)
            self.register_buffer(name=f'mask_{i}', tensor=idx, persistent=False)
            # m[0].weight.data[~idx] = 0.0

    def set_kernnel(self):
        r = self.range // 2
        width = r // self.k
        cx, cy = torch.meshgrid(torch.linspace(start=-r, end=r, steps=self.range),
                                torch.linspace(start=-r, end=r, steps=self.range))
        dist = torch.sqrt(cx * cx + cy * cy)
        for i, m in enumerate(self.collect):
            idx = (dist < (r - i * width)) & (dist > (r - i * width - width))
            idx = einops.repeat(idx, 's1 s2 -> n c s1 s2', n=m[0].weight.shape[0], c=1)
            m[0].weight.data[~idx] = 0.0
        print('Set NLU.')

    def forward(self, x):
        ret = 0
        for i, (su, coll) in enumerate(zip(self.subunit, self.collect)):
            if self.training:
                coll[0].weight.data[~eval(f'self.mask_{i}')] = 0.0
            ret += coll(su(x))
        ret = ret / self.k
        return ret


class YChannel(nn.Module):
    def __init__(self, ch=64, ks_crf=7, ks_ncrf=21, k=5):
        super(YChannel, self).__init__()
        self.k1 = ks_crf
        self.k2 = ks_ncrf

        self.CRF = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ks_crf, padding=ks_crf // 2, groups=ch, bias=False),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )

        self.inh_w = nn.Parameter(torch.Tensor([0.]))
        self.nCRF = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ks_ncrf, padding=ks_ncrf // 2, groups=ch,
                      bias=False),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )

        self.nlu = NLU(ncrf_size=ks_ncrf, ch=ch, k=k)
        self.combine = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1)

    def regist_mask(self):
        r = self.k2 // 2
        cx, cy = torch.meshgrid(torch.linspace(start=-r, end=r, steps=self.k2),
                                torch.linspace(start=-r, end=r, steps=self.k2))
        dist = torch.sqrt(cx * cx + cy * cy)
        idx = (dist < r) & (dist > (self.k1 // 2) * 1.414)
        idx = einops.repeat(idx, 's1 s2 -> n c s1 s2', n=self.nCRF[0].weight.shape[0], c=1)
        self.register_buffer(name='mask', tensor=idx, persistent=False)

    def set_kernnel(self):
        r = self.k2 // 2
        cx, cy = torch.meshgrid(torch.linspace(start=-r, end=r, steps=self.k2),
                                torch.linspace(start=-r, end=r, steps=self.k2))
        dist = torch.sqrt(cx * cx + cy * cy)
        idx = (dist < r) & (dist > (self.k1 // 2) * 1.414)
        idx = einops.repeat(idx, 's1 s2 -> n c s1 s2', n=self.nCRF[0].weight.shape[0], c=1)
        self.nCRF[0].weight.data[~idx] = 0.0
        print('Set Y nCRF.')

    def forward(self, x):
        r1 = self.CRF(x)
        if self.training:
            self.nCRF[0].weight.data[~self.mask] = 0.0
        r2 = self.nCRF(x)
        r3 = self.nlu(x)
        r = self.combine(r1 + r3 - self.inh_w * r2)
        return r


class SimpleCell(nn.Module):
    def __init__(self, ch=64):
        super(SimpleCell, self).__init__()
        self.or1 = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 5), padding=(1, 2), bias=False, groups=ch),
            nn.InstanceNorm2d(num_features=ch),
            # 一个channel内做归一化，算H*W的均值，用在风格化迁移。在图像风格化中，生成结果主要依赖于某个图像实例
            # 所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
            nn.ReLU(inplace=True)
        )
        self.or2 = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(5, 3), padding=(2, 1), bias=False, groups=ch),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )
        self.or3 = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=5, padding=2, bias=False, groups=ch),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )
        self.or4 = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=5, padding=2, bias=False, groups=ch),
            nn.InstanceNorm2d(num_features=ch),
            nn.ReLU(inplace=True)
        )
        self.combine = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1)

    def regist_mask(self):
        cx, cy = torch.meshgrid(torch.linspace(start=-2, end=2, steps=5), torch.linspace(start=-2, end=2, steps=5))
        idx1 = ((cx + cy) <= 1) & ((cx + cy) >= -1)
        idx2 = ((cx - cy) <= 1) & ((cx - cy) >= -1)
        idx1 = einops.repeat(idx1, 's1 s2 -> n c s1 s2', n=self.or3[0].weight.shape[0], c=1)
        idx2 = einops.repeat(idx2, 's1 s2 -> n c s1 s2', n=self.or4[0].weight.shape[0], c=1)
        self.register_buffer(name='mask_1', tensor=idx1, persistent=False)
        self.register_buffer(name='mask_2', tensor=idx2, persistent=False)

    def set_kernnel(self):
        cx, cy = torch.meshgrid(torch.linspace(start=-2, end=2, steps=5), torch.linspace(start=-2, end=2, steps=5))
        idx1 = ((cx + cy) <= 1) & ((cx + cy) >= -1)
        idx2 = ((cx - cy) <= 1) & ((cx - cy) >= -1)
        idx1 = einops.repeat(idx1, 's1 s2 -> n c s1 s2', n=self.or3[0].weight.shape[0], c=1)
        idx2 = einops.repeat(idx2, 's1 s2 -> n c s1 s2', n=self.or4[0].weight.shape[0], c=1)
        self.or3[0].weight.data[~idx1] = 0
        self.or4[0].weight.data[~idx2] = 0
        print('Set SimpleCell.')

    def forward(self, x):
        o1 = self.or1(x)
        o2 = self.or2(x)
        if self.training:
            self.or3[0].weight.data[~self.mask_1] = 0
            self.or4[0].weight.data[~self.mask_2] = 0
        o3 = self.or3(x)
        o4 = self.or4(x)
        res = self.combine(o1 + o2 + o3 + o4)
        return res


class adap_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(adap_conv, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x


class Combine(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(Combine, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel)
        self.factor = factor
        # self.conv1 = nn.Conv2d(in_channels=out_channel, out_channels=3, kernel_size=1)
        # self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=3, kernel_size=1)
        # self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)


        if self.factor >= 2:
            self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel),
                                              requires_grad=require_grad)
        self.CDC1 = CDC(channel=out_channel)
        self.CDC2 = CDC(channel=out_channel)

        self.stage_CSAC = CSAC(channel=10)
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=out_channel, kernel_size=3, padding=1)
    def forward(self, *input):
        _, _, H, W = input[0].size()
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        if self.factor >= 2:
            x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor / 2),
                                    output_padding=(
                                    x1.size(2) - x2.size(2) * self.factor, x1.size(3) - x2.size(3) * self.factor))



        cd1 = self.CDC1(x1)
        cd2 = self.CDC2(x2)
        cs1 = self.stage_CSAC(cd1)
        cs2 = self.stage_CSAC(cd2)
        x1 = self.conv1(cs1)
        x2 = self.conv2(cs2)

        return x1+x2

class CDC(nn.Module):
    def __init__(self, channel):
        super(CDC, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // 3, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(channel // 3, 10, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channel // 3, 10, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(channel // 3, 10, kernel_size=7, padding=3)
        self.conv9 = nn.Conv2d(channel // 3, 10, kernel_size=9, padding=4)

    def forward(self, x):
        y = self.conv1(x)
        x3 = self.conv3(y)
        x5 = self.conv5(y)
        x7 = self.conv7(y)
        x9 = self.conv9(y)
        return x3 + x5 + x7 + x9

class CSAC(nn.Module):
    def __init__(self, channel):
        super(CSAC, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, padding=0, bias=0)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


class dcode(nn.Module):
    def __init__(self):
        super(dcode, self).__init__()
        self.CDC1 = CDC(channel=32)
        self.CDC2 = CDC(channel=32)
        self.CDC3 = CDC(channel=32)
        # self.CDC4 = CDC(channel=32)
        self.stage_CSAC = CSAC(channel=10)  # 输入输出channels一致
        self.fuse = nn.Conv2d(3, 1, kernel_size=1, padding=0)
        self.conv1_1 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        _, _, H, W = x[0].size()
        # print("1")
        cd1 = self.CDC1(x[0])
        # cd1 = x[0]
        # print(cd1.shape)
        cd2 = self.CDC2(x[1])
        cd3 = self.CDC3(x[2])
        # cd4 = self.CDC4(x[3])

        cs1 = self.stage_CSAC(cd1)
        cs2 = self.stage_CSAC(cd2)
        cs3 = self.stage_CSAC(cd3)
        # cs4 = self.stage_CSAC(cd4)
        # print(cs3.shape)

        y1 = self.conv1_1(cs1)
        y2 = self.conv1_1(cs2)
        y3 = self.conv1_1(cs3)
        # y4 = self.conv1_1(cs4)
        # y1 = F.interpolate(y1, [H, W], mode="bilinear", align_corners=False)
        y2 = F.interpolate(y2, [H, W], mode="bilinear", align_corners=False)
        y3 = F.interpolate(y3, [H, W], mode="bilinear", align_corners=False)
        # y4 = F.interpolate(y4, [H, W], mode="bilinear", align_corners=False)

        y = self.fuse(torch.cat([y1, y2, y3], dim=1)).sigmoid()
        return y, y1.sigmoid(), y2.sigmoid(), y3.sigmoid()

class VisualNet(nn.Module):
    def __init__(self):
        super(VisualNet, self).__init__()
        ch = 32
        self.expand = nn.Conv2d(in_channels=3, out_channels=ch, kernel_size=3, padding=1)
        self.xc = XChannel(ch=ch, ks_crf=7, ks_ncrf=21)
        # self.xc = BasicBlock(ch=ch, ks_crf=7, ks_ncrf=21)

        self.yc = YChannel(ch=ch, ks_crf=7, ks_ncrf=21)
        # self.yc = BasicBlock(ch=ch, ks_crf=7, ks_ncrf=21)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.simple1 = SimpleCell(ch=ch)
        self.simple2 = SimpleCell(ch=ch)



        self.combine2 = Combine(in_channel=(ch, ch), out_channel=ch, factor=2)
        self.combine3 = Combine(in_channel=(ch, ch), out_channel=16, factor=2)
        self.head = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
        for m in self.modules():
            if hasattr(m, 'regist_mask'):
                m.regist_mask()

    def forward(self, image):


        x = self.expand(image)
        # print(x.shape)
        # _, _, H, W = x.size()

        xres = self.xc(x)
        yres = self.yc(x)
        x0 = xres + yres

        x1 = self.mp(x0)
        x2 = self.simple1(x1)

        x3 = self.mp(x2)
        x4 = self.simple2(x3)

        #
        cx1 = self.combine2(x2, x4)
        cx2 = self.combine3(x0, cx1)

        ret = self.head(cx2)

        return ret.sigmoid()
            # ,x0.sigmoid(),cx1.sigmoid(),cx2.sigmoid()


if __name__ == '__main__':
    net = VisualNet().cuda().train()
    x = torch.randn(1, 3, 150, 150).cuda()
    y = net(x)
    print(y[0].shape)
    # torch.save(net.state_dict(), './test.pth')
    # for k, v in net.state_dict().items():
    #     print(k)

    # state1 = torch.load('./test.pth')
    # state2 = torch.load('../001-bsds/3model.pth')
    # for (k1, v1), (k2, v2) in zip(state1.items(), state2.items()):
    #     if k1 == k2:
    #         print(k1, v1.numel() - v2.numel())
    #     else:
    #         print('/t', k1)
    #         print('/t', k2)

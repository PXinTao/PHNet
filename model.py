import math
import os
from typing import List

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tcm

import mstr
import utils


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        _, _, h, w = x.shape
        y_embed, x_embed = torch.meshgrid(torch.linspace(1, h, h, device=x.device), torch.linspace(1, w, w, device=x.device))
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[None, :, :, None] / dim_t
        pos_y = y_embed[None, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos + x

def pad(input, patch_size):
    _, _, h, w = input.shape
    p_h = patch_size - (h % patch_size)
    p_w = patch_size - (w % patch_size)
    x = F.pad(input, pad=(p_w//2, p_w//2+p_w%2, p_h//2, p_h//2+p_h%2))
    return x

def crop(x, shape):
    _, _, h, w = shape
    _, _, _h, _w = x.shape
    p_h = (_h - h) // 2 + 1
    p_w = (_w - w) // 2 + 1
    return x[:, :, p_h:p_h+h, p_w:p_w+w]

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class VGG16Reducer(nn.Module):
    def __init__(self, kernel_size, out_channels):
        super(VGG16Reducer, self).__init__()
        self.sub_scale1_1 = nn.Conv2d(in_channels= 64, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        self.sub_scale1_2 = nn.Conv2d(in_channels= 64, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)

        self.sub_scale2_1 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        self.sub_scale2_2 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)

        self.sub_scale3_1 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        self.sub_scale3_2 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        self.sub_scale3_3 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)

        self.sub_scale4_1 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        self.sub_scale4_2 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        self.sub_scale4_3 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)

        self.sub_scale5_1 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        self.sub_scale5_2 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        self.sub_scale5_3 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)

        # self.scale1 = nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        # self.scale2 = nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        # self.scale3 = nn.Conv2d(in_channels=3*out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        # self.scale4 = nn.Conv2d(in_channels=3*out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)
        # self.scale5 = nn.Conv2d(in_channels=3*out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1)

        self.unsample2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size= 4, stride=2, bias=False)
        self.unsample3 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size= 8, stride=4, bias=False)
        self.unsample4 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=16, stride=8, bias=False)
        self.unsample5 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=16, stride=8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(get_upsampling_weight(in_channels=out_channels, out_channels=out_channels, kernel_size=m.kernel_size[0]))
                m.weight.requires_grad = False

    def forward(self, side_output):
        [conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3] = side_output
        s1_1 = self.sub_scale1_1(conv1_1)
        s1_2 = self.sub_scale1_2(conv1_2)
        s2_1 = self.sub_scale2_1(conv2_1)
        s2_2 = self.sub_scale2_2(conv2_2)
        s3_1 = self.sub_scale3_1(conv3_1)
        s3_2 = self.sub_scale3_2(conv3_2)
        s3_3 = self.sub_scale3_3(conv3_3)
        s4_1 = self.sub_scale4_1(conv4_1)
        s4_2 = self.sub_scale4_2(conv4_2)
        s4_3 = self.sub_scale4_3(conv4_3)
        s5_1 = self.sub_scale5_1(conv5_1)
        s5_2 = self.sub_scale5_2(conv5_2)
        s5_3 = self.sub_scale5_3(conv5_3)

        s1 = sum([s1_1, s1_2])
        s2 = self.unsample2(sum([s2_1, s2_2]))
        s3 = self.unsample3(sum([s3_1, s3_2, s3_3]))
        s4 = self.unsample4(sum([s4_1, s4_2, s4_3]))
        s5 = self.unsample5(sum([s5_1, s5_2, s5_3]))

        return [s1, s2, s3, s4, s5]


class VGG16(nn.Module):
    def __init__(self, cfgs):
        super(VGG16, self).__init__()
        # VGG16
        self.conv1_1 = nn.Conv2d(3,  64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64,  64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        # self.pool4 = nn.Identity()
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)

        if cfgs['pretrain']:
            model_paramters = torch.load(cfgs['vgg16-5stage'])
            for k, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    if "funnel" not in k:
                        m.weight.data = model_paramters.popitem(last=False)[-1]
                        m.bias.data = model_paramters.popitem(last=False)[-1]
                    else:
                        nn.init.normal_(m.weight.data, 0.0, 0.02)
                        nn.init.constant_(m.bias.data, 0.0)

    def forward(self, image):
        conv1_1 = self.relu1_1(self.conv1_1(image))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))

        pool1 = self.pool1(conv1_2)
        conv2_1 = self.relu2_1(self.conv2_1(pool1))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))

        pool2 = self.pool2(conv2_2)
        conv3_1 = self.relu3_1(self.conv3_1(pool2))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))

        pool3 = self.pool3(conv3_3)
        conv4_1 = self.relu4_1(self.conv4_1(pool3))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))

        pool4 = self.pool4(conv4_3)
        conv5_1 = self.relu5_1(self.conv5_1(pool4))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))

        return [conv1_2, conv2_2, conv3_3, conv4_3, conv5_3]


class DepthWisePatchEmbedding(nn.Sequential):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(DepthWisePatchEmbedding, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=patch_size, stride=patch_size, groups=in_channels, bias=False),
            nn.InstanceNorm2d(num_features=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=1)
        )


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


class Refine_block2_1(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(Refine_block2_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel)
        self.factor = factor
        if self.factor >= 2:
            self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)

    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        if self.factor >= 2:
            x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
        return x1 + x2


class MSFuse(nn.Module):
    def __init__(self, cfgs):
        super(MSFuse, self).__init__()
        embed_dim = 512
        self.patch_embeddings = nn.ModuleList(
            [
                DepthWisePatchEmbedding(in_channels=ic, patch_size=32//2**i, embed_dim=embed_dim)
                for i, ic in enumerate([64, 128, 256, 512, 512])
            ]
        )
        self.msatt = nn.ModuleList(
            [
                mstr.Block(dim=embed_dim, n_heads=8, mlp_ratio=2)
                for _ in range(3)
            ]
        )

        self.t5 = Refine_block2_1(in_channel=(512, 512), out_channel=512, factor= 1*2)
        self.t4 = Refine_block2_1(in_channel=(512, 512), out_channel=512, factor= 2*2)
        self.t3 = Refine_block2_1(in_channel=(256, 512), out_channel=256, factor= 4*2)
        self.t2 = Refine_block2_1(in_channel=(128, 512), out_channel=128, factor= 8*2)
        self.t1 = Refine_block2_1(in_channel=( 64, 512), out_channel= 64, factor=16*2)

        self.f54 = Refine_block2_1(in_channel=(512, 512), out_channel=256, factor=2)
        self.f43 = Refine_block2_1(in_channel=(256, 256), out_channel=128, factor=2)
        self.f32 = Refine_block2_1(in_channel=(128, 128), out_channel= 64, factor=2)
        self.f21 = Refine_block2_1(in_channel=( 64,  64), out_channel= 32, factor=2)
        self.collect = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

        # self.collect5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1), nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2**5, stride=2**4, bias=False))
        # self.collect4 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1), nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2**4, stride=2**3, bias=False))
        # self.collect3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1), nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2**3, stride=2**2, bias=False))
        # self.collect2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1), nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2**2, stride=2**1, bias=False))
        # self.collect1 = nn.Conv2d(in_channels= 64, out_channels=1, kernel_size=1)
        # self.collect0 = nn.Conv2d(in_channels=  5, out_channels=1, kernel_size=1)

        # self.deconv_weight5 = nn.Parameter(utils.bilinear_upsample_weights(2**4, 1), requires_grad=False)
        # self.deconv_weight4 = nn.Parameter(utils.bilinear_upsample_weights(2**3, 1), requires_grad=False)
        # self.deconv_weight3 = nn.Parameter(utils.bilinear_upsample_weights(2**2, 1), requires_grad=False)
        # self.deconv_weight2 = nn.Parameter(utils.bilinear_upsample_weights(2**1, 1), requires_grad=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                # if m.bias is not None:
                #     nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(get_upsampling_weight(in_channels=1, out_channels=1, kernel_size=m.kernel_size[0]))
                m.weight.requires_grad = False
        self.features = VGG16(cfgs)

    def forward(self, images):
        shape = images.shape
        images = pad(images, patch_size=32)
        sides = self.features(images)

        side_embeddings = []
        for s, embd in zip(sides, self.patch_embeddings):
            side_embeddings.append(embd(s))
        side_embeddings = torch.stack(side_embeddings, dim=1)

        side_embeddings = einops.rearrange(side_embeddings, 'n nt d h w -> n h w nt d')
        for m in self.msatt:
            side_embeddings = m(side_embeddings)
        side_embeddings = einops.rearrange(side_embeddings, 'n h w nt d -> n nt d h w')
        side_embeddings = torch.split(side_embeddings, split_size_or_sections=1, dim=1)
        
        s5 = self.t5(sides[4], side_embeddings[4].squeeze(1))
        s4 = self.t4(sides[3], side_embeddings[3].squeeze(1))
        s3 = self.t3(sides[2], side_embeddings[2].squeeze(1))
        s2 = self.t2(sides[1], side_embeddings[1].squeeze(1))
        s1 = self.t1(sides[0], side_embeddings[0].squeeze(1))

        s4 = self.f54(s4, s5)
        s3 = self.f43(s3, s4)
        s2 = self.f32(s2, s3)
        s1 = self.f21(s1, s2)

        x = crop(self.collect(s1), shape)
        return x.sigmoid()



class Cross_Entropy(nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()
        # self.weight1 = nn.Parameter(torch.Tensor([1.]))
        # self.weight2 = nn.Parameter(torch.Tensor([1.]))

    def forward(self, pred, labels, side_output=None):
        # def forward(self, pred, labels):
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]
        side_output = side_output
        total_loss = cross_entropy_per_image(pred, labels)
        if side_output is not None:
            for s in side_output:
                total_loss += cross_entropy_per_image(s, labels) / len(side_output)

        # total_loss = cross_entropy_per_image(pred, labels)
        # total_loss = dice_loss_per_image(pred, labels)
        # total_loss = 1.00 * cross_entropy_per_image(pred, labels) + \
        # 0.00 * 0.1 * dice_loss_per_image(pred, labels)
        # total_loss = self.weight1.pow(-2) * cross_entropy_per_image(pred, labels) + \
        #              self.weight2.pow(-2) * 0.1 * dice_loss_per_image(pred, labels) + \
        #              (1 + self.weight1 * self.weight2).log()
        return total_loss, (1-pred_pos).abs(), pred_neg


def dice(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    dice = ((logits * labels).sum() * 2 + eps) / \
        (logits.sum() + labels.sum() + eps)
    dice_loss = dice.pow(-1)
    return dice_loss


def dice_loss_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += dice(_logit, _label)
    return total_loss / len(logits)


def cross_entropy_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += cross_entropy_orignal(_logit, _label)
    return total_loss / len(logits)


def cross_entropy_orignal(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-5
    pred_pos = logits[labels > 0].clamp(eps, 1.0 - eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)

    w_anotation = labels[labels > 0]

    weight_pos, weight_neg = get_weight(labels, labels, 0.17, 2.5)

    cross_entropy = (-pred_pos.log() * weight_pos * w_anotation).sum() + \
        (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy


def cross_entropy_with_weight(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > 0].clamp(eps, 1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0-eps)
    w_anotation = labels[labels > 0]
    # weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)
    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    # cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
    #                     (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy


def get_weight(src, mask, threshold, weight):
    count_pos = src[mask > threshold].size()[0]
    count_neg = src[mask == 0.0].size()[0]
    total = count_neg + count_pos
    weight_pos = count_neg / total
    weight_neg = (count_pos / total) * weight
    return weight_pos, weight_neg


def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * \
            (decay_rate ** (epoch // decay_steps))


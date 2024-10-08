import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss


class UnetFormerLoss_DetialLoss(nn.Module):
    def __init__(self, ignore_index=255, in_dim=128):
        super().__init__()
        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.detial_loss = DetailAggregateLoss(in_dim=in_dim)


    def forward(self, logits, labels):
        if self.training and len(logits) == 2 and isinstance(logits, (list, tuple)):
            logit_main, logit_detial = logits
            loss = self.main_loss(logit_main, labels) + 0.7 * self.detial_loss(logit_detial, labels)
        else:
            loss = self.main_loss(logits, labels)

        return loss


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class DetialOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(DetialOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()


    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


class DetailAggregateLoss(nn.Module):
    def __init__(self, in_dim=128):
        super(DetailAggregateLoss, self).__init__()
        self.out = DetialOutput(in_dim, 64, 1).cuda()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(
            torch.cuda.FloatTensor)
        # a b c 
        # fuse_kernel k1*a+k2*b+k3*c
        self.fuse_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]],\
                         dtype=torch.float32).reshape(1, 3, 1, 1). \
                type(torch.cuda.FloatTensor))


    def forward(self, boundary_logits, gtmasks):
        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_logits = self.out(boundary_logits)
        if len(gtmasks.shape) != 4:
            gtmasks = gtmasks.unsqueeze(1)

        boundary_targets = F.conv2d(gtmasks.type(torch.cuda.FloatTensor),
                                    self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.type(torch.cuda.FloatTensor),
                                       self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.type(torch.cuda.FloatTensor),
                                       self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.type(torch.cuda.FloatTensor),
                                       self.laplacian_kernel, stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:],
                                               mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:],
                                               mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:],
                                               mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack(
            (boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss + dice_loss


    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            nowd_params += list(module.parameters())
        return nowd_params

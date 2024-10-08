import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from geoseg.models.CSwin import CSWinT
from geoseg.models.ScaleLayerASPP import SLASPP
from geoseg.models.sfam import SFAM
from geoseg.models.vit import vit_tiny
from geoseg.models.iformer import iformer_small
from einops import rearrange


class SIM(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(SIM, self).__init__()
        sem_channels = in_channels2
        out_channels = in_channels2
        self.up = nn.ConvTranspose2d(in_channels1, sem_channels, kernel_size=2, stride=2)
        self.local_cv = nn.Sequential(nn.Conv2d(sem_channels, sem_channels, 1),
                                      nn.BatchNorm2d(sem_channels))
        self.global_cv1 = nn.Sequential(nn.Conv2d(in_channels2, sem_channels, 1),
                                        nn.BatchNorm2d(sem_channels),
                                        nn.Sigmoid())
        self.global_cv2 = nn.Sequential(nn.Conv2d(in_channels2, sem_channels, 1),
                                        nn.BatchNorm2d(sem_channels))
        self.conv = DoubleConv(sem_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = self.global_cv1(x2) * self.local_cv(x1) + self.global_cv2(x2)
        x = self.conv(x)
        return x


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(nn.Conv2d(in_channels, num_classes, kernel_size=1))


class UnetDecoder(nn.Module):
    def __init__(self, detial_loss=False, encoder_channels=[64, 128, 256, 512], class_num=6):
        super(UnetDecoder, self).__init__()
        encoder_channels = list(reversed(encoder_channels))
        self.detial_loss = detial_loss
        self.up1 = Up(encoder_channels[0], encoder_channels[1])
        self.up2 = Up(encoder_channels[1], encoder_channels[2])
        self.up3 = Up(encoder_channels[2], encoder_channels[3])
        self.final_conv = nn.Conv2d(encoder_channels[3], class_num, kernel_size=1, padding=0)

    def forward(self, x, h, w, fm=False):
        x2, x3, x4, x5 = x
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        if fm:
            return x
        x = self.final_conv(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return (x, x3) if self.detial_loss else x


class SIMDecoder(nn.Module):
    def __init__(self, encoder_channels=[64, 128, 256, 512], msadd=False, class_num=6):
        super(SIMDecoder, self).__init__()
        encoder_channels = list(reversed(encoder_channels))
        self.up1 = SIM(encoder_channels[0], encoder_channels[1])
        self.up2 = SIM(encoder_channels[1], encoder_channels[2])
        self.up3 = SIM(encoder_channels[2], encoder_channels[3])
        self.msadd = msadd
        if self.msadd:
            self.eac = nn.Sequential(nn.Conv2d(64 * 15, encoder_channels[3], 1, 1, 0),
                                     nn.BatchNorm2d(encoder_channels[3]),
                                     nn.ReLU6(inplace=True))

        self.final_conv = nn.Conv2d(encoder_channels[3], class_num, kernel_size=1, padding=0)

    def forward(self, x, h, w, fm=False):
        if not self.msadd:
            x2, x3, x4, x5 = x
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.final_conv(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            return x
        else:
            x2, x3, x4, x5 = x
            x4 = self.up1(x5, x4)
            x3 = self.up2(x4, x3)
            x2 = self.up3(x3, x2)
            x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=False)
            x5 = F.interpolate(x5, size=(h, w), mode='bilinear', align_corners=False)
            x = torch.cat([x2, x3, x4, x5], 1)
            x = self.eac(x)
            x = self.final_conv(x)
            return x


class SFAMDecoder(nn.Module):
    def __init__(self, encoder_channels=[64, 128, 256, 512], detial_loss=False, class_num=6):
        super(SFAMDecoder, self).__init__()
        encoder_channels = list(reversed(encoder_channels))
        self.detial_loss = detial_loss
        self.up1 = SFAM(encoder_channels[0], encoder_channels[1], encoder_channels[0])
        self.up2 = SFAM(encoder_channels[1], encoder_channels[2], encoder_channels[1])
        self.up3 = Up(encoder_channels[2], encoder_channels[3])
        self.final_conv = nn.Conv2d(encoder_channels[3], class_num, kernel_size=1, padding=0)

    def forward(self, x, h, w, fm=False):
        x2, x3, x4, x5 = x
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        if fm:
            return x
        x = self.final_conv(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return (x, x3) if self.detial_loss else x


class HyCSUnet(nn.Module):
    def __init__(self, encoder='unet', decoder='unet', slaspp=False, fcm=False, msadd=False,
                 detial_loss=False, num_classes=6, w_o_dw=False, aff=False, low_pass=False, only_hp=False,
                 only_aff=False):
        super(HyCSUnet, self).__init__()
        assert encoder in ['unet', 'cswin', 'iformer']
        assert decoder in ['unet', 'sim', 'sfam']

        if encoder == 'unet':
            self.encoder = timm.create_model('resnet18', features_only=True, output_stride=32,
                                             out_indices=(1, 2, 3, 4))
        elif encoder == 'iformer':
            self.encoder = iformer_small(
                pretrained=True if not w_o_dw else False, w_o_dw=w_o_dw, aff=aff, low_pass=low_pass, only_hp=only_hp,
                only_aff=only_aff)
        elif encoder == 'cswin':
            self.encoder = CSWinT(fcm=fcm)

        if decoder == 'unet':
            self.decoder = UnetDecoder(detial_loss=detial_loss, class_num=num_classes)
        elif decoder == 'sim':
            self.decoder = SIMDecoder(class_num=num_classes, msadd=msadd)
        elif decoder == 'sfam':
            self.decoder = SFAMDecoder(class_num=num_classes, detial_loss=detial_loss)

        self.slaspp = slaspp

        if self.slaspp:
            self.slaspp1 = SLASPP(512)
            self.slaspp2 = SLASPP(256)

    def forward(self, x, fm=False, lhf_fm=0):
        h, w = x.size()[-2:]
        x = self.encoder(x, lhf_fm)
        if lhf_fm in [2, 3, 4]:
            return x
        if self.slaspp:
            x2, x3, x4, x5 = x
            x5 = self.slaspp1(x5)
            x4 = self.slaspp2(x4)
            x = x2, x3, x4, x5
        x = self.decoder(x, h, w, fm)
        return x


class IASD_DBB(nn.Module):
    # iformer_aff_sfam_dl.py with different backbone
    def __init__(self, encoder: str = 'unet', decoder='sfam', slaspp=False, fcm=False, msadd=False,
                 detial_loss=False, num_classes=6, w_o_dw=False, aff=False, low_pass=False, only_hp=False,
                 only_aff=False):
        super(IASD_DBB, self).__init__()
        self.fms = [None]
        assert encoder in ['resnet18', 'vitt', 'swint']
        assert decoder == 'sfam'

        if encoder == 'resnet18':
            self.encoder = timm.create_model(encoder, features_only=True, output_stride=32,
                                             out_indices=(1, 2, 3, 4))
        elif encoder == 'vitt':
            # timm.models
            self.encoder = vit_tiny()

        elif encoder == 'swint':
            self.encoder = timm.create_model('swin_tiny_patch4_window7_224', output_stride=32, features_only=True,
                                             img_size=512, window_size=8, out_indices=(0, 1, 2, 3))

        encoder_channels = [64, 128, 256, 512] if encoder != 'swint' else [96, 192, 384, 768]
        self.decoder = SFAMDecoder(encoder_channels=encoder_channels, class_num=num_classes, detial_loss=detial_loss)
        self.encoder_name = encoder
        self.slaspp = slaspp

        if self.slaspp:
            self.slaspp1 = SLASPP(512)
            self.slaspp2 = SLASPP(256)

    def forward(self, x, fm=False, lhf_fm=0):
        h, w = x.size()[-2:]
        x = self.encoder(x)
        if self.encoder_name == 'swint':
            x = [rearrange(i, 'b h w c -> b c h w') for i in x]
        if fm:
            return x[0]  # TODO /16
        if self.slaspp:
            x2, x3, x4, x5 = x
            x5 = self.slaspp1(x5)
            x4 = self.slaspp2(x4)
            x = x2, x3, x4, x5
        x = self.decoder(x, h, w)
        return x

    def register_gene_fm(self, tls):
        for tl in tls:
            tl.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.fms[0] = output.cpu().detach()


if __name__ == '__main__':
    m = FCM((64, 64), 256)
    im = torch.randn(2, 4096, 256)
    y = m(im)
    if isinstance(y, (tuple, list)):
        for x in y:
            print(x.shape)
    else:
        print(y.shape)

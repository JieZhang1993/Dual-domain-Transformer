import torch.nn as nn
import torch
import torch.nn.functional as F

class SLAM(nn.Module):
    def __init__(self, out_channels):
        super(SLAM, self).__init__()
        self.cv = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.bn(self.cv(x)))
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SLAM(out_channels)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SLAM(out_channels))

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class SLASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18]):
        super(SLASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SLAM(out_channels)
        ))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for i, conv in enumerate(self.convs):
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

if __name__ == '__main__':
    im = torch.randn(2, 256, 128, 128)
    model = SLASPP(256)
    y = model(im)
    print(y.shape)

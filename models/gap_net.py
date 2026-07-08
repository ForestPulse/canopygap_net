import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.GroupNorm(8, out_channels)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual_conv(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.InstanceNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.InstanceNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi[0].bias.data.fill_(0.1)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi + x

class Sentinel2ResUNet(nn.Module):
    def __init__(self, in_channels=220, s1_in_channels=3, base_channels=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.s2_stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
        )

        self.s1_stem = nn.Sequential(
            nn.Conv2d(s1_in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse0 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
        )

        self.enc1a = ResidualUNetBlock(base_channels, 64, kernel_size=3)
        self.enc1b = ResidualUNetBlock(base_channels, 64, kernel_size=7)

        self.enc2a = ResidualUNetBlock(64, 128, kernel_size=3)
        self.enc2b = ResidualUNetBlock(64, 128, kernel_size=7)

        self.enc3a = ResidualUNetBlock(128, 256, kernel_size=3)
        self.enc3b = ResidualUNetBlock(128, 256, kernel_size=7)

        self.bottleneck = ResidualUNetBlock(256, 512)

        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualUNetBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualUNetBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualUNetBlock(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, s2, s1):
        """
        s2: (B, 220, 256, 256)
        s1: (B,   3, 256, 256)
        """
        s2f = self.s2_stem(s2)
        s1f = self.s1_stem(s1)

        x0 = self.fuse0(torch.cat([s2f, s1f], dim=1))

        e1a = self.enc1a(x0)
        e1b = self.enc1b(x0)
        e1 = e1a + e1b

        p1 = self.pool(e1)
        e2a = self.enc2a(p1)
        e2b = self.enc2b(p1)
        e2 = e2a + e2b

        p2 = self.pool(e2)
        e3a = self.enc3a(p2)
        e3b = self.enc3b(p2)
        e3 = e3a + e3b

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        e3g = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3g], dim=1))

        d2 = self.up2(d3)
        e2g = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2g], dim=1))

        d1 = self.up1(d2)
        e1g = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1g], dim=1))

        return self.final(d1)
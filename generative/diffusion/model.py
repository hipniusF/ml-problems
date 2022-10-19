from asyncore import readwrite
from urllib.parse import uses_params
from xml.dom.minidom import Identified
import torch
from torch import nn
from torch import functional as F
import numpy as np
from einops import rearrange
from tqdm import tqdm, trange


class Downsample(nn.Module):
    def __init__(self, in_c, out_c):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        return self.act(x)


class Upsample(nn.Module):
    def __init__(self, in_c, out_c):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return self.act(x)


class Attention(nn.Module):
    def __init__(self, in_c, num_heads=8):
        super(Attention, self).__init__()
        self.c = in_c  # // 8
        self.num_heads = num_heads
        self.to_vkq = nn.Conv2d(in_c, num_heads*self.c*3, 1)
        self.to_out = nn.Conv2d(self.c*num_heads, in_c, 1)

    def forward(self, x):
        size = x.size()
        vqk = self.to_vkq(x).view(
            size[0], 3, self.num_heads, self.c, size[-1], size[-2])
        v = vqk[:, 0]
        q = vqk[:, 1]
        k = vqk[:, 2]

        h = torch.einsum('b h c x y, b h c z y -> b h c x z', v, k)
        h /= size[-1]*size[-2]
        h = h.softmax(dim=-1)

        h = torch.einsum('b h c x y, b h c z y -> b h c x z', h, q)
        h = rearrange(h, 'b h c x y -> b (c h) x y')
        
        return self.to_out(h)



class Block(nn.Module):
    def __init__(self, in_c, out_c, groups=8):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(groups, out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=8):
        super(ResidualBlock, self).__init__()
        self.block1 = Block(in_c, in_c, groups if in_c % groups == 0 else in_c)
        self.block2 = Block(in_c, out_c, groups if out_c %
                            groups == 0 else out_c)
        self.conv = nn.Conv2d(
            in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.conv(x)


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        channels = [3, 128, 256, 512, 512]
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        attent_dims = [1024, 512, 256]

        c_pairs = list(zip(channels[:-1], channels[1:]))
        for i, (in_c, out_c) in enumerate(c_pairs):
            self.down_blocks.append(nn.ModuleList(
                (ResidualBlock(in_c, out_c), ResidualBlock(out_c, out_c), Downsample(
                    out_c, out_c), Attention(out_c) if in_c in attent_dims else nn.Identity())
            ))
        for i, (out_c, in_c) in enumerate(reversed(c_pairs)):
            self.up_blocks.append(nn.ModuleList(
                (ResidualBlock(2*in_c, in_c), ResidualBlock(2*in_c, out_c),
                 Upsample(in_c, in_c), Attention(in_c) if in_c in attent_dims else nn.Identity())
            ))

        self.middle_conv = nn.Conv2d(channels[-1], channels[-1], 3, padding=1)

    def forward(self, x):
        h = []
        for block1, block2, down, attn in self.down_blocks:
            x = block1(x)
            h.append(x)

            x = attn(x)

            x = block2(x)
            h.append(x)

            x = down(x)

        x = self.middle_conv(x)

        for block1, block2, up, attn in self.up_blocks:
            x = up(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)

            x = attn(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x)
        return x


class DenoisingDiffusion(nn.Module):
    def __init__(self, diffusion_steps=100):
        super(DenoisingDiffusion, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.betas = self._get_betas('linear')
        self.unet = Unet()

    def _get_betas(self, mode):
        if mode == 'linear':
            return np.linspace(1e-4, .02, self.diffusion_steps)

    @torch.no_grad()
    def forward_process(self, img):
        alphas = torch.tensor(1 - self.betas)
        for t in trange(self.diffusion_steps):
            alpha_bar = torch.prod(alphas[:t])
            img = torch.sqrt(alpha_bar)*img \
                + torch.sqrt(1-alpha_bar)*torch.randn(img.size())
        return img

    def p_sample(self, x_t, t):
        return self.unet(x_t)

    def backward_process(self, x_t, steps):
        for t in steps:
            x_t = self.p_sample(x_t, t)
        return x_t

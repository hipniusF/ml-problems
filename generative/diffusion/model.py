from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch import functional as F
from einops import rearrange
from tqdm.notebook import tqdm, trange
from ema_pytorch import EMA

from matplotlib import pyplot as plt


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoder(nn.Module):
    def __init__(self, c, length):
        super(PositionalEncoder, self).__init__()
        self.dim = c
        self.pe = self.__get_pe(c, length).to(get_device())
        self.lin = nn.Linear(c, c)

    def __get_pe(self, d_model, length):
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))

        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(np.log(10000.0) / d_model)))

        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe
    
    def forward(self, x, t):
        return x + rearrange(self.lin(self.pe[t]), 'd -> d 1 1')


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
    def __init__(self, in_c, out_c, t_length, groups=8):
        super(ResidualBlock, self).__init__()
        self.block1 = Block(in_c, in_c, groups if in_c % groups == 0 else in_c)
        self.block2 = Block(in_c, out_c, groups if out_c %
                            groups == 0 else out_c)
        self.conv = nn.Conv2d(
            in_c, out_c, 1) if in_c != out_c else nn.Identity()
        
        self.pos_enc = PositionalEncoder(in_c, t_length) if in_c % 2 == 0 else None

    def forward(self, x, t):
        if (self.pos_enc is not None):
            x = self.pos_enc(x, t)
        h = self.block1(x)
        h = self.block2(h)
        return h + self.conv(x)


class Unet(nn.Module):
    def __init__(self, t_length):
        super(Unet, self).__init__()
        channels = [3, 128, 256, 512, 512]
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        attent_dims = [512, 512, 256]

        c_pairs = list(zip(channels[:-1], channels[1:]))
        for i, (in_c, out_c) in enumerate(c_pairs):
            self.down_blocks.append(nn.ModuleList(
                (ResidualBlock(in_c, out_c, t_length), ResidualBlock(out_c, out_c, t_length), Downsample(
                    out_c, out_c), Attention(out_c) if in_c in attent_dims else nn.Identity())
            ))
        for i, (out_c, in_c) in enumerate(reversed(c_pairs)):
            self.up_blocks.append(nn.ModuleList(
                (ResidualBlock(2*in_c, in_c, t_length), ResidualBlock(2*in_c, out_c, t_length),
                 Upsample(in_c, in_c), Attention(in_c) if in_c in attent_dims else nn.Identity())
            ))

        self.middle_conv = nn.Conv2d(channels[-1], channels[-1], 3, padding=1)

    def forward(self, x, t):
        h = []
        for block1, block2, down, attn in self.down_blocks:
            x = block1(x, t)
            h.append(x)

            x = attn(x)

            x = block2(x, t)
            h.append(x)

            x = down(x)

        x = self.middle_conv(x)

        for block1, block2, up, attn in self.up_blocks:
            x = up(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = attn(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
        return x


class DenoisingDiffusion(nn.Module):
    def __init__(self, diffusion_steps=100, train=False, dev=get_device()):
        super(DenoisingDiffusion, self).__init__()
        self.dev = dev
        self.diffusion_steps = diffusion_steps
        self.betas = self.__get_betas('linear')
        self.alphas = torch.tensor(1 - self.betas)
        self.alpha_bars = [torch.prod(self.alphas[:t])
                            for t, _ in enumerate(self.alphas)]
        self.unet = Unet(diffusion_steps)
        if train:
            self.__init_train()
        self.to(dev)

    def __init_train(self):
        self.epoch = 0
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=6e-4)
        self.losses = []
        self.ema = EMA(
            self.unet,
            beta=.9999,
            update_after_step = 100,
            update_every=100
        ).to(self.dev)

    def __get_betas(self, mode):
        if mode == 'linear':
            return np.linspace(1e-4, .02, self.diffusion_steps)

    def q_sample(self, x_t, t):
        return torch.sqrt(self.alpha_bars[t])*x_t \
            + torch.sqrt(1-self.alpha_bars[t])*torch.randn(x_t.size())

    @torch.no_grad()
    def forward_process(self, img):
        for t in trange(self.diffusion_steps):
            img = self.q_sample(img, t)
        return img

    def predict_eps(self, x_t, t):
        return self.unet(x_t, t)

    def p_sample(self, x_t, t):
            z = torch.randn(x_t.size()).to(self.dev) if t > 1 else 0
            eps_hat = self.predict_eps(x_t, t)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            predicted_error = (1-alpha)/(1-alpha_bar)*eps_hat
            return 1/torch.sqrt(alpha) * (x_t - predicted_error) + self.betas[t]*z

    @torch.no_grad()
    def backward_process(self, x_t, steps=None):
        if steps is None:
            steps = self.diffusion_steps
        x_t = x_t.to(self.dev)
        for t in trange(1, steps):
            x_t = self.p_sample(x_t, t)
            
        return x_t

    def train_loop(self, epochs, loader):
        self.train()

        for self.epoch in trange(self.epoch + 1, epochs):
            for x_0, _ in tqdm(loader, leave=False):
                x_0 = x_0.to(self.dev)
                t = np.random.randint(1, self.diffusion_steps)
                eps = torch.randn(x_0.size(), device=self.dev)
                alpha_bar = self.alpha_bars[t]
                eps_hat = self.predict_eps(torch.sqrt(alpha_bar)*x_0 + torch.sqrt(1-alpha_bar)*eps, t)

                for p in self.parameters():
                    p.grad = None
                loss = self.loss_fn(eps, eps_hat)
                loss.backward()
                self.optim.step()
                self.ema.update()
                self.losses.append(loss.item())

            self.save(f'./chkpnts/checkpnt_epoch-{self.epoch}.pt')
            self.notify(self.backward_process(torch.randn((1, 3, 128, 128)).to(self.dev)))
    
    def notify(self, x):
        '''
        Send notification through:
        https://github.com/marcoperg/telegram-notifier
        '''

        import os
        import io
        import requests
        import dotenv
        dotenv.load_dotenv()

        x = x.squeeze()
        print(x.shape)
        x = x.movedim((1, 2, 0), (0, 1, 2))
        x = x.detach().cpu().numpy()
        x = (x + 1) / 2
        x = np.clip(x, 0, 1)
        plt.imshow(x)
        plt.show()

        buf = io.BytesIO()
        plt.imsave(buf, x, format='png')
        image_data = buf.getvalue()
        url = 'http://localhost:3000'
        files = {'photo': image_data}
        headers = {'token': os.environ['SECRET']}
        data = {'text': f'Epoch {self.epoch}'}
        requests.post(url, files=files, data=data, headers=headers)
        
    def save(self, path):
            torch.save({'net': self.state_dict(), 
                        'optim': self.optim.state_dict(),
                        'losses': self.losses,
                        'epoch': self.epoch,
                        'ema': self.ema.state_dict(),
                        'timestamp': str(datetime.now())
                       },
                path)
    
    def load(self, path):
        chk = torch.load(path)
        self.load_state_dict(chk['net'])
        self.optim.load_stat_dict(chk['optim'])
        self.ema.load_state_dict(chk['ema'])
        self.losses = chk['losses']
        self.epoch = chk['epoch']
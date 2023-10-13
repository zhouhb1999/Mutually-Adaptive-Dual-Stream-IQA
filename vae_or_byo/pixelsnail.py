"""
    Based heavily off this implementation: https://github.com/rosinality/vq-vae-2-pytorch/blob/master/pixelsnail.py

    Changes:
        - Some naming conventions are changed (don't use `input` as a variable!!!)
        - support for $n$ conditioning variables.
        - conditioning stack as recommended by original authors (rather than guesswork)
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torchvision
import torchvision.transforms.functional as VF 
from torchvision import transforms

from math import sqrt, prod
from functools import partial, lru_cache

from helper import HelperModule

class WNConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self = torch.nn.utils.weight_norm(self)
        self.forward = super().forward

class CausalConv2d(HelperModule):
    def build(self, in_channel, out_channel, kernel_size, stride=1, padding='downright'):
        assert padding in ['downright', 'down', 'causal'], f"Unknown padding type! Got '{padding}'"

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]
        elif padding in ['down', 'causal']:
            pad = [kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] - 1, 0]
        self.causal = kernel_size[1] // 2 if padding == 'causal' else 0
        self.pad = nn.ZeroPad2d(pad)
        self.conv = WNConv2d(in_channel, out_channel, kernel_size, stride=stride)

    def forward(self, x):
        x = self.pad(x)
        if self.causal:
            self.conv.weight_v.data[:, :, -1, self.causal:].zero_()
        x = self.conv(x)
        return x

class GatedResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, conv='wnconv2d', dropout=0.1, condition_dim=0, aux_channels=0):
        super().__init__()

        # TODO: change conv keywords to something simpler
        assert conv in ['wnconv2d', 'causal_downright', 'causal'], "Invalid conv argument [wnconv2d, causal_downright, causal]"
        if conv == 'wnconv2d':
            conv_builder = partial(WNConv2d, padding=kernel_size // 2)
        elif conv == 'causal_downright':
            conv_builder = partial(CausalConv2d, padding='downright')
        elif conv == 'causal':
            conv_builder = partial(CausalConv2d, padding='causal')

        self.conv1 = conv_builder(in_channel, channel, kernel_size)
        self.conv2 = conv_builder(channel, in_channel*2, kernel_size)
        # self.conv2 = conv_builder(channel, in_channel, kernel_size)
        self.drop1 = nn.Dropout(dropout)

        if aux_channels > 0:
            self.aux_conv = WNConv2d(aux_channels, channel, 1)

        if condition_dim > 0:
            self.convc = WNConv2d(condition_dim, in_channel*2, 1, bias=False)
            # self.convc = WNConv2d(condition_dim, in_channel, 1, bias=False)
            # self.alphac = nn.Parameter(torch.tensor(0.0))
        # self.alpha = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.GLU(1) 

    def forward(self, x, a=None, c=None):
        y = self.conv1(F.elu(x))

        if a != None:
            y = y + self.aux_conv(F.elu(a))
        y = F.elu(y)

        y = self.drop1(y)
        y = self.conv2(y)

        if c != None and len(c) > 0:
            y = self.convc(c) + y
        y = self.gate(y) + x
        return y
        # return self.alpha * F.elu(y) + x 

@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        torch.from_numpy(mask).unsqueeze(0),
        torch.from_numpy(start_mask).unsqueeze(1),
    )

class CausalAttention(HelperModule):
    def build(self, query_channel, key_channel, channel, nb_heads=8, dropout=0.1):
        self.to_qkv = nn.Linear(
                        query_channel + key_channel + key_channel,
                        channel*3
                    )
        self.to_qkv = torch.nn.utils.weight_norm(self.to_qkv)

        self.head_dim = channel // nb_heads
        self.nb_heads = nb_heads

        self.drop = nn.Dropout(dropout)

    def forward(self, q, k):
        batch_size, _, height, width = k.shape
        reshape = lambda x: x.view(batch_size, -1, self.nb_heads, self.head_dim).transpose(1, 2)

        qkv_f = torch.cat([
            q.view(batch_size, q.shape[1], -1).transpose(1, 2),
            k.view(batch_size, k.shape[1], -1).transpose(1, 2),
            k.view(batch_size, k.shape[1], -1).transpose(1, 2),
        ], dim=-1)

        q, k, v = torch.chunk(self.to_qkv(qkv_f), chunks=3, dim=-1)
        q, k, v = reshape(q), reshape(k).transpose(2, 3), reshape(v)

        attn = (q @ k) / sqrt(self.head_dim)
        mask, start_mask = causal_mask(height*width)
        mask, start_mask = mask.type_as(q), start_mask.type_as(q)

        attn = attn.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(attn, dim=3) * start_mask
        attn = self.drop(attn)

        y = (attn @ v).transpose(1, 2).reshape(batch_size, height, width, self.head_dim*self.nb_heads).permute(0, 3, 1, 2)
        return y

class PixelBlock(nn.Module):
    def __init__(self, 
            in_channel, 
            channel, 
            kernel_size, 
            nb_res_blocks,
            attention = True,
            dropout = 0.1, 
            condition_dim = 0):
        super().__init__()
        self.res_blks = nn.ModuleList([
            GatedResBlock(in_channel, channel, kernel_size, conv='causal', dropout=dropout, condition_dim=condition_dim) 
            for _ in range(nb_res_blocks)
        ])

        self.attention = attention
        if self.attention:
            self.k_blk = GatedResBlock(in_channel*2 + 2, in_channel, 1, dropout=dropout)
            self.q_blk = GatedResBlock(in_channel + 2, in_channel, 1, dropout=dropout)
            self.attn = CausalAttention(in_channel + 2, in_channel*2 + 2, in_channel // 2, dropout=dropout)
            self.out_blk = GatedResBlock(in_channel, in_channel, 1, dropout=dropout, aux_channels=in_channel // 2)
        else:
            self.out_blk = WNConv2d(in_channel + 2, in_channel, 1)

    def forward(self, x, bg, c=None):
        y = x
        for blk in self.res_blks:
            y = blk(y, c=c)

        if self.attention:
            k = self.k_blk(torch.cat([x, y, bg], 1))
            q = self.q_blk(torch.cat([y, bg], 1))
            y = self.out_blk(y, a=self.attn(q, k))
        else:
            y = self.out_blk(torch.cat([y, bg], dim=1))

        return y

# TODO: Rename this to something more generic(?)
class CondResNet(HelperModule):
    def build(self, in_channel, channel, kernel_size, nb_res_blocks):
        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]
        blocks.extend([
            GatedResBlock(channel, channel, kernel_size)
            for _ in range(nb_res_blocks)
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class PixelSnail(nn.Module):
    def __init__(self, 
            shape, 
            nb_class, 
            channel, 
            kernel_size, 
            nb_pixel_block, 
            nb_res_block, 
            res_channel, 
            dropout = 0.1, 
            nb_cond = 0,
            nb_cond_res_block = 0,
            nb_cond_in_res_block = 0,
            cond_embedding_dim = 64,
            cond_res_channel = 0, 
            cond_res_kernel = 3, 
            nb_out_res_block = 0,
            attention = True,
        ):
        super().__init__()
        height, width = shape
        self.nb_class = nb_class

        assert kernel_size % 2, "Kernel size must be odd"

        # avoids blind spot issue in original PixelCNN
        self.horz_conv = CausalConv2d(nb_class, channel, [kernel_size // 2, kernel_size], padding='down')
        self.vert_conv = CausalConv2d(nb_class, channel, [(kernel_size+1) // 2, kernel_size // 2], padding='downright')

        # builds coordinate embeddings
        coord_x = torch.arange(-0.5, 0.5, 1 / height)
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = torch.arange(-0.5, 0.5, 1 / width)
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)

        self.register_buffer('bg', torch.cat([coord_x, coord_y], 1).half()) 

        # defines list of PixelBlocks
        self.blks = nn.ModuleList([
            PixelBlock(channel, res_channel, kernel_size, nb_res_block, dropout=dropout, condition_dim=cond_res_channel, attention=attention) 
        for _ in range(nb_pixel_block)])
        
        # if we have conditioning variables, build conditioning stack
        if nb_cond > 0:
            # combined conditons resnet
            self.cond_net = CondResNet(nb_cond*cond_embedding_dim, cond_res_channel, cond_res_kernel, nb_cond_res_block)
            # create smaller conditioning resnet for all but the largest condition
            self.cond_in_net = nn.ModuleList([
                CondResNet(cond_embedding_dim, cond_embedding_dim, cond_res_kernel, nb_cond_in_res_block) if nb_cond_in_res_block > 0 else nn.Identity()
            for _ in range(nb_cond - 1)])
            self.cond_in_net.append(nn.Identity())

        self.nb_cond = nb_cond

        # create output residual stack
        out = []
        for _ in range(nb_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))
        out.append(nn.ELU(inplace=True))
        out.append(WNConv2d(channel, nb_class, 1))

        self.out = nn.Sequential(*out)

        self.shift_down = lambda x, size=1: F.pad(x, [0,0,size,0])[:, :, :x.shape[2], :]
        self.shift_right = lambda x, size=1: F.pad(x, [size,0,0,0])[:, :, :, :x.shape[3]]

    # one hot encode function to avoid any explicit casts to float / half
    def _one_hot(self, x: torch.LongTensor):
        batch, height, width = x.shape
        y = torch.zeros(batch, self.nb_class, height, width).to(x.device)
        y[torch.arange(batch).view(-1, 1, 1), x] = 1
        return y

    # cache is used to increase speed of sampling
    def forward(self, x, cs = None, cache = None):
        if cache is None:
            cache = {}
        batch, height, width = x.shape
        y = self._one_hot(x)
       
        horz = self.shift_down(self.horz_conv(y))
        vert = self.shift_right(self.vert_conv(y))
        y = horz + vert

        bg = self.bg[:, :, :height, :].expand(batch, 2, height, width)

        if cs != None and len(cs) > 0:
            if 'condition' in cache:
                cs = cache['condition']
                cs = cs[:, :, :height, :]
            else:
                cs = [self.cond_in_net[i](c) for i, c in enumerate(cs)] # apply smaller resnet where appropriate
                cs = [F.interpolate(c, size=(height, width)) for c in cs] # interpolate conditions to image size
                cs = torch.cat(cs, dim=1) # concatenate conditions
                cs = self.cond_net(cs) # apply larger conditional resnet
                cache['condition'] = cs.detach().clone()
                cs = cs[:, :, :height, :]

        # iterate over pixelblocks
        for blk in self.blks:
            y = blk(y, bg, c=cs)
        y = self.out(y)
        return y, cache

if __name__ == "__main__":
    ps = PixelSnail([6, 6], 512, 64, 7, 7, 2, 32, cond_res_channel=32, nb_out_res_block=5)
    x = torch.LongTensor(1, 6, 6).random_(0, 255)
    print(ps(x)[0].shape)

    x = torch.LongTensor(1, 24, 24).random_(0, 255)
    cs = [torch.LongTensor(1, 6, 6).random_(0, 255), torch.LongTensor(1, 12, 12).random_(0, 255)]
    ps = PixelSnail([24, 24], 512, 64, 7, 7, 2, 32, nb_cond_res_block=3, cond_res_channel=32, nb_out_res_block=5, nb_cond=2)
    print(ps(x, cs=cs)[0].shape)

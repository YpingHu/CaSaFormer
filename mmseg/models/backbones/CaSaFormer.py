import math
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.runner import _load_checkpoint
from mmseg.utils import get_root_logger

from ..builder import BACKBONES


def _make_divisible(v, divisor, min_value=None):
    """
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            ks: int,
            stride: int,
            expand_ratio: int,
            activations=None,
            norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks // 2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LightFeaturePyramidModule(nn.Module):
    def __init__(
            self,
            cfgs,
            out_indices,
            inp_channel=16,
            activation=nn.ReLU,
            norm_cfg=dict(type='BN', requires_grad=True),
            width_mult=1.):
        super().__init__()
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),
            activation()
        )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)  # 保证通道数能被8整除
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg,
                                     activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class Attention(torch.nn.Module):
    def __init__(self, dim, dim_s, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim_s, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim_s, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x, singlex):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        _, C_s, H_s, W_s = get_shape(singlex)

        qq1 = self.to_q(singlex)
        qq2 = qq1.reshape(B, self.num_heads, self.key_dim, H_s * W_s)
        qq = qq2.permute(0, 1, 3, 2)
        kk1 = self.to_k(x)
        kk = kk1.reshape(B, self.num_heads, self.key_dim, H * W)
        vv1 = self.to_v(x)
        vv2 = vv1.reshape(B, self.num_heads, self.d, H_s * W_s)
        vv = vv2.permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class Block(nn.Module):

    def __init__(self, dim, dim_s, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(dim, dim_s=dim_s, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                              activation=act_layer, norm_cfg=norm_cfg)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim_s * mlp_ratio)
        self.mlp = Mlp(in_features=dim_s, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, singlex, x=None):
        if x is not None:
            singlex = singlex + self.drop_path(self.attn(x, singlex))
            singlex = singlex + self.drop_path(self.mlp(singlex))
        else:
            singlex = singlex + self.drop_path(self.attn(singlex, singlex))
            singlex = singlex + self.drop_path(self.mlp(singlex))
        return singlex


class CaSaBlock(nn.Module):
    def __init__(self,
                 channels,
                 depths=4,
                 key_dim=16,
                 num_heads=8,
                 mlp_ratios=2,
                 attn_ratios=2,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6):
        super().__init__()

        self.channels = channels
        self.embed_dim = sum(channels)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.cross_trans = nn.ModuleList()
        self.self_trans = nn.ModuleList()
        for i in range(depths):
            self.cross_trans.append(Block(
                self.embed_dim, self.channels[i], key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratios, attn_ratio=attn_ratios,
                drop=0, drop_path=dpr[i] if isinstance(dpr, list) else dpr,
                norm_cfg=norm_cfg,
                act_layer=act_layer))
            self.self_trans.append(Block(
                self.channels[i], self.channels[i], key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratios, attn_ratio=attn_ratios,
                drop=0, drop_path=dpr[i] if isinstance(dpr, list) else dpr,
                norm_cfg=norm_cfg,
                act_layer=act_layer))

    def forward(self, xx):
        x = xx.split(self.channels, dim=1)
        results = []
        for i in range(len(self.channels)):
            semantics = self.self_trans[i](self.cross_trans[i](x[i], xx))
            results.append(semantics)
        result = torch.cat(results, dim=1)
        return result


class CrossAttentionGatedFuse(torch.nn.Module):
    '''
    Fuse high-level and low-level futures through cross-attention block.
    '''
    def __init__(self, dim, dim_s, out_dim, key_dim, num_heads,
                 attn_ratio=2, drop_path=0.,
                 act_layer=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 max_len=64**2,
                 device='cuda:0'):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim_s, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(act_layer(), Conv2d_BN(
            self.dh, dim_s, bn_weight_init=0, norm_cfg=norm_cfg))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.conv = ConvModule(dim_s, out_dim, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)

        # positional encoding
        pe_l = torch.zeros((max_len, dim_s), device=device)
        pe_h = torch.zeros((max_len, dim), device=device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_s, 2) *
                             -(math.log(10000.0) / dim_s))
        pe_l[:, 0::2] = torch.sin(position * div_term)
        pe_l[:, 1::2] = torch.cos(position * div_term)
        self.pe_l = pe_l.unsqueeze(0)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim))
        pe_h[:, 0::2] = torch.sin(position * div_term)
        pe_h[:, 1::2] = torch.cos(position * div_term)
        self.pe_h = pe_h.unsqueeze(0)

    def forward(self, low, high):
        B, C, H, W = get_shape(high)
        _, C_l, H_l, W_l = get_shape(low)

        low_pe = self.pe_l[:, :H_l * W_l, :]
        low = low + low_pe.permute(0, 2, 1).reshape(1, C_l, H_l, W_l)
        high_pe = self.pe_h[:, :H * W, :]
        high2 = high + high_pe.permute(0, 2, 1).reshape(1, C, H, W)

        qq = self.to_q(low).reshape(B, self.num_heads, self.key_dim, H_l * W_l).permute(0, 1, 3, 2)
        kk = self.to_k(high2).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(high).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)  # V不需要positional encoding

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H_l, W_l)
        low_weight = self.proj(xx)

        upsample_high = F.interpolate(high, size=(H_l, W_l), mode='bilinear', align_corners=False)

        fuse_f = low_weight * low + upsample_high
        fuse_f = self.conv(fuse_f)

        return fuse_f

@BACKBONES.register_module()
class CaSaFormer(BaseModule):
    def __init__(self, cfgs,
                 channels,
                 out_channels,
                 embed_out_indice,
                 decode_out_indices=[1, 2, 3],
                 num_casablocks=2,
                 depths=4,
                 key_dim=16,
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=2,
                 c2t_stride=2,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6,
                 init_cfg=None,
                 injection=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.injection = injection
        self.embed_dim = sum(channels)
        self.decode_out_indices = decode_out_indices
        self.init_cfg = init_cfg
        self.num_casablocks = num_casablocks
        if self.init_cfg != None:
            self.pretrained = self.init_cfg['checkpoint']
        else:
            self.pretrained = None

        self.lfpm = LightFeaturePyramidModule(cfgs=cfgs, out_indices=embed_out_indice, norm_cfg=norm_cfg)
        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.fuse_trans = nn.ModuleList()
        for i in range(depths):
            if i in self.decode_out_indices:
                self.fuse_trans.append(CrossAttentionGatedFuse(
                    self.channels[i], self.channels[i], out_dim=self.out_channels[i], key_dim=key_dim,
                    num_heads=num_heads, attn_ratio=attn_ratios, drop_path=dpr[i] if isinstance(dpr, list) else dpr,
                    norm_cfg=norm_cfg, act_layer=act_layer, ))
            else:
                self.fuse_trans.append(None)

        self.casa_blocks = nn.ModuleList()
        for i in range(num_casablocks):
            self.casa_blocks.append(CaSaBlock(self.channels, depths=depths, key_dim=key_dim, num_heads=num_heads,
                mlp_ratios=mlp_ratios, attn_ratios=attn_ratios, drop_path_rate=drop_path_rate,norm_cfg=norm_cfg,
                act_layer=act_layer))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict_ema' in checkpoint:
                state_dict = checkpoint['state_dict_ema']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        outputs = self.lfpm(x)
        out = self.ppa(outputs)

        for i in range(self.num_casablocks):
            out = self.casa_blocks[i](out)

        results = []
        semantics = out.split(self.channels, dim=1)
        for i in range(len(self.channels)):
            if i in self.decode_out_indices:
                out_ = self.fuse_trans[i](outputs[i], semantics[i])
                results.append(out_)

        return results




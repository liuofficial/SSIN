import os
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

from model.involution import Involution


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 16, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'invo_S': _cfg(crop_pct=0.9),
    'invo_S': _cfg(crop_pct=0.9),
    'invo_S': _cfg(crop_pct=0.875),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvoMLP(nn.Module):
    def __init__(self, dim, ikernel, group, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight1 = Mlp(dim, dim // 4, dim * 2)
        self.reweight2 = Mlp(dim, dim // 4, dim * 2)

        self.proj = nn.Linear(dim, dim)
        self.projcnn = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.invo = Involution(kernel_size=ikernel, in_channel=dim, stride=1, group=group)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.i = ikernel
        self.g = group
    def forward(self, x):
        B, H, W, C = x.shape

        hw = self.invo(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        c = self.mlp_c(x)

        a = (hw + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight1(a).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = hw * a[0] + c * a[1]

        x = self.projcnn(x.permute(0,3,1,2)).permute(0,2,3,1)

        return x


class InvoBlock(nn.Module):

    def __init__(self, dim, ikernel, group, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=InvoMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, ikernel, group, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbedOverlapping(nn.Module):
    """ 2D Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None, groups=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size
        # remove image_size in model init to support dynamic image size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    """ Downsample transition stage
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def basic_blocks(dim, ikernel, group, index, layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0., skip_lam=1.0, mlp_fn=InvoMLP, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(InvoBlock(dim, ikernel, group, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.Sequential(*blocks)

    return blocks


class InvoNet(nn.Module):
    def __init__(self, layers, ikernel, group, inc=96, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.LayerNorm, mlp_fn=InvoMLP, fork_feat=False):

        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbedOverlapping(patch_size=1, stride=1, padding=0, in_chans=inc, embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], ikernel, group, i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break

            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size))

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """ mmseg or mmdet `init_weight` """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)

        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        outs = []
        mid_outs = []

        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        if self.fork_feat:
            return outs

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x

        x = self.norm(x).permute(0,2,1).reshape(B,C,H,W)

        return x


@register_model
def InvoMLP_B(inc, ikernel, group, pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2]
    mlp_ratios = [4]
    embed_dims = [inc]
    model = InvoNet(layers, ikernel, group, inc=inc, embed_dims=embed_dims, patch_size=1, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=InvoMLP, **kwargs)
    model.default_cfg = default_cfgs['invo_S']
    return model



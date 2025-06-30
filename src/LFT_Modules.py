# Based on FasterViT

from torch import nn, Tensor
from typing import Optional
import torch, copy,math
import torch.nn as nn
from timm.models.layers import DropPath, LayerNorm2d
import numpy as np
from src.transformer import TransformerDecoder, TransformerDecoderLayer,TransformerEncoderLayer
from .BAT_Modules import BoundaryWiseAttentionGate2D, BoundaryWiseAttentionGate1D, BoundaryWiseAttentionGateAtrous2D, BoundaryWiseAttentionGateAtrous1D

class LightweightFusionTransformer(nn.Module):
    def __init__(self,
                 point_pred_layers=6,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6, #12
                 num_decoder_layers=2,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False,
                 return_intermediate_dec=False,
                 BAG_type='2D',
                 Atrous=False,
                 dim=128,
                 depth=6,
                 input_resolution=int(32), #32
                 num_heads=8,
                 window_size=16,
                 ct_size = 1,
                 conv = False,
                 downsample = True,
                 mlp_ratio = 4.,
                 qkv_bias = True,
                 qk_scale = None,
                 drop = 0.,
                 attn_drop = 0.,
                 drop_path = 0.,
                 layer_scale = None,
                 layer_scale_conv = None,
                 only_local = False,
                 hierarchy = True,
                 do_propagation = True):
        super().__init__()
        sr_ratio = input_resolution // window_size if not only_local else 1
        self.window_size = window_size

        encoder_layer = LightweightFusionTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            BAG_type=BAG_type,
            Atrous=Atrous,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            dim=dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            sr_ratio=sr_ratio,
            window_size=window_size,
            layer_scale=layer_scale,
            ct_size=ct_size,
            do_propagation=do_propagation
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = LightweightFusionTransformerEncoder(point_pred_layers,
                                                       encoder_layer,
                                                       num_encoder_layers,
                                                       encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead,
                                                dim_feedforward, dropout,
                                                activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        # (b, 128, 32, 32)
        # print("TTTinsrc", src.shape)

        # src = src.flatten(2).permute(2, 0, 1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory, weights = self.encoder(src,
                                       src_key_padding_mask=mask,
                                       pos=pos_embed,
                                       height=h,
                                       width=w)
        # (1,128,32,32)
        # print("memory",memory.shape)
        memory = memory.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # print("memory2222", memory.shape)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt,
                          memory,
                          memory_key_padding_mask=mask,
                          pos=pos_embed,
                          query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h,
                                                                w), weights


class LightweightFusionTransformerEncoder(nn.Module):
    def __init__(self,
                 point_pred_layers,
                 encoder_layer,
                 num_layers,
                 norm=None,
                 dim = 128,
                 depth = 6,
                 input_resolution = int(32),  # 32
                 num_heads = 8,
                 window_size = 16,
                 ct_size = 1,
                 conv = False,
                 downsample = True,
                 mlp_ratio = 4.,
                 qkv_bias = True,
                 qk_scale = None,
                 drop = 0.,
                 attn_drop = 0.,
                 drop_path = 0.,
                 layer_scale = None,
                 layer_scale_conv = None,
                 only_local = False,
                 hierarchy = True,
                 do_propagation = True):
        super().__init__()
        self.transformer_block = True
        self.window_size = window_size
        if not only_local and input_resolution // window_size > 1 and hierarchy :
            self.global_tokenizer = TokenInitializer(dim,
                                                     input_resolution,
                                                     window_size,
                                                     ct_size=ct_size)
            self.do_gt = True
        else:
            self.do_gt = False

        self.point_pred_layers = point_pred_layers
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                height: int = 32,
                width: int = 32):
        weights = []
        ct = self.global_tokenizer(src) if self.do_gt else None
        B, C, H, W = src.shape
        if self.transformer_block:
            output = window_partition(src, self.window_size)
        for layer_i, layer in enumerate(self.layers):
            if layer_i > self.num_layers - self.point_pred_layers - 1:
                output, ct,weight = layer(
                    True,
                    False,
                    output,
                    ct,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos,
                    height=height,
                    width=width,
                    b_size=B)
                # print("weight",weight.shape)
                weights.append(weight)
                # print("append-weight", weight.shape)
            else:
                output, ct = layer(
                    False,
                    True,
                    output,
                    ct,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos,
                    height=height,
                    width=width,
                    b_size=B)

        if self.transformer_block:
            output = window_reverse(output, self.window_size, H, W, B)

        if self.norm is not None:
            output = self.norm(output)

        return output, weights



class LightweightFusionTransformerEncoderLayer(nn.Module):
    """
    Using HAT Lightweight Improvement Transformer's Encoder
    Add Boundary-wise Attention Gate to Transformer's Encoder
    """

    def __init__(self,
                 d_model,
                 nhead,
                 BAG_type='2D',
                 Atrous=True,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False,
                 dim=128,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1.,
                 window_size=16,
                 last=False,
                 layer_scale=None,
                 ct_size=2,
                 do_propagation=True):
        super().__init__()
        self.normalize_before = normalize_before
        # positional encoding for windowed attention tokens
        self.pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=window_size**2)
        self.norm1 = norm_layer(dim)
        # number of carrier tokens per every window
        cr_tokens_per_window = ct_size**2 if sr_ratio > 1 else 0
        # total number of carrier tokens
        cr_tokens_total = cr_tokens_per_window*sr_ratio*sr_ratio
        self.cr_window = ct_size
        self.attn = WindowAttention(dim,
                                    num_heads=nhead,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    resolution=window_size,
                                    seq_length=window_size**2 + cr_tokens_per_window)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.window_size = window_size

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # if do hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = WindowAttention(
                dim,
                num_heads=nhead, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, resolution=int(cr_tokens_total**0.5),
                seq_length=cr_tokens_total)

            self.hat_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.hat_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.hat_pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=cr_tokens_total)
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            self.upsampler = nn.Upsample(size=window_size, mode='nearest')

        # keep track for the last block to explicitly add carrier tokens to feature maps
        self.do_propagation = do_propagation

        if BAG_type == '1D':
            if Atrous:
                self.BAG = BoundaryWiseAttentionGateAtrous1D(d_model)
            else:
                self.BAG = BoundaryWiseAttentionGate1D(d_model)
        elif BAG_type == '2D':
            if Atrous:
                self.BAG = BoundaryWiseAttentionGateAtrous2D(d_model)
            else:
                self.BAG = BoundaryWiseAttentionGate2D(d_model)
        self.BAG_type = BAG_type

    def forward(self,
                use_bag,
                last,
                src,
                carrier_tokens,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                height: int = 32,
                width: int = 32,
                b_size: int=1):
        # print("b_size = ",b_size)
        # src = (b^2,256,128)
        B, T, N = src.shape
        # print("b_size = ",b_size)
        ct = carrier_tokens
        x = self.pos_embed(src)

        if self.sr_ratio > 1:
            # do hierarchical attention via carrier tokens
            # first do attention for carrier tokens
            Bg, Ng, Hg = ct.shape

            # ct are located quite differently
            ct = ct_dewindow(ct, self.cr_window*self.sr_ratio, self.cr_window*self.sr_ratio, self.cr_window)

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)

            # attention plus mlp
            ct = ct + self.hat_drop_path(self.gamma1*self.hat_attn(self.hat_norm1(ct)))
            ct = ct + self.hat_drop_path(self.gamma2*self.hat_mlp(self.hat_norm2(ct)))

            # ct are put back to windows
            ct = ct_window(ct, self.cr_window * self.sr_ratio, self.cr_window * self.sr_ratio, self.cr_window)

            ct = ct.reshape(x.shape[0], -1, N)
            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)

        # window attention together with carrier tokens
        x = x + self.drop_path(self.gamma3*self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4*self.mlp(self.norm2(x)))

        if self.sr_ratio > 1:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split([x.shape[1] - self.window_size*self.window_size, self.window_size*self.window_size], dim=1)
            ct = ctr.reshape(Bg, Ng, Hg) # reshape carrier tokens.
            if last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(B, N, self.cr_window, self.cr_window)
                x = x + self.gamma1 * self.upsampler(ctr_image_space.to(dtype=torch.float32)).flatten(2).transpose(1, 2).to(dtype=x.dtype)
        # return x, ct

        if self.normalize_before:
            # features = self.forward_pre(src, src_mask, src_key_padding_mask,
            #                             pos)
            if use_bag:
                b, c = x.shape[1:]
                if self.BAG_type == '1D':
                    features = x.permute(1, 2, 0)
                    features, weights = self.BAG(features)
                    features = features.permute(2, 0, 1).contiguous()
                    weights = weights.view(b, 1, height, width)
                elif self.BAG_type == '2D':
                    features = x.permute(1, 2, 0).view(b, c, height, width)
                    features, weights = self.BAG(features)
                    features = features.flatten(2).permute(2, 0, 1).contiguous()
                return features,ct,weights
            else:
                return x,ct

        # features = self.forward_post(src, src_mask, src_key_padding_mask, pos)
        # (1024,b,128)
        # print("TTTfeatures",x.shape)
        if use_bag:
            features = window_reverse(x,self.window_size,32,32,b_size)
            # (1,128,32,32)
            # print("BAGinfeatures",features.shape)
            b, c = features.shape[2:]
            if self.BAG_type == '1D':
                features = x.permute(1, 2, 0)
                features, weights = self.BAG(features)
                features = features.permute(2, 0, 1).contiguous()
                weights = weights.view(b, 1, height, width)
            elif self.BAG_type == '2D':

                features, weights = self.BAG(features)

            # (1024,b,128)
            # print("outTfeatures",features.shape)

            features = window_partition(features,self.window_size)
            # (4,256,128)
            # print("BAGoutfeatures",x.shape)

            return features,ct, weights
        else:
            return x,ct



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def window_partition(x, window_size):
    # x = (1, 128, 32, 32)  //(b,c,h,w)
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    # windows = (4, 256, 128)  // ( (h/ window_size)*(w / window_size), window_size*window_size, c)
    return windows


def window_reverse(windows, window_size, H, W, B):
    # windows = (4, 256, 128)  //( (h/ window_size)*(w / window_size), window_size*window_size, c)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    # x = (1, 128, 32, 32)  //(b,c,h,w)
    return x


def ct_dewindow(ct, W, H, window_size):
    bs = ct.shape[0]
    N=ct.shape[2]
    ct2 = ct.view(-1, W//window_size, H//window_size, window_size, window_size, N).permute(0, 5, 1, 3, 2, 4)
    ct2 = ct2.reshape(bs, N, W*H).transpose(1, 2)
    return ct2


def ct_window(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct = ct.view(bs, H // window_size, window_size, W // window_size, window_size, N)
    ct = ct.permute(0, 1, 3, 2, 4, 5)
    return ct

class PosEmbMLPSwinv2D(nn.Module):
    def __init__(self,
                 window_size,
                 pretrained_window_size,
                 num_heads, seq_length,
                 ct_correct=False,
                 no_log=False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        self.seq_length = seq_length
        self.register_buffer("relative_bias", relative_bias)
        self.ct_correct=ct_correct

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor, local_window_size):
        if self.deploy:
            input_tensor += self.relative_bias
            return input_tensor
        else:
            self.grid_exists = False

        if not self.grid_exists:
            self.grid_exists = True

            relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            n_global_feature = input_tensor.shape[2] - local_window_size
            if n_global_feature > 0 and self.ct_correct:

                step_for_ct=self.window_size[0]/(n_global_feature**0.5+1)
                seq_length = int(n_global_feature ** 0.5)
                indices = []
                for i in range(seq_length):
                    for j in range(seq_length):
                        ind = (i+1)*step_for_ct*self.window_size[0] + (j+1)*step_for_ct
                        indices.append(int(ind))

                top_part = relative_position_bias[:, indices, :]
                lefttop_part = relative_position_bias[:, indices, :][:, :, indices]
                left_part = relative_position_bias[:, :, indices]
            relative_position_bias = torch.nn.functional.pad(relative_position_bias, (n_global_feature,
                                                                                      0,
                                                                                      n_global_feature,
                                                                                      0)).contiguous()
            if n_global_feature>0 and self.ct_correct:
                relative_position_bias = relative_position_bias*0.0
                relative_position_bias[:, :n_global_feature, :n_global_feature] = lefttop_part
                relative_position_bias[:, :n_global_feature, n_global_feature:] = top_part
                relative_position_bias[:, n_global_feature:, :n_global_feature] = left_part

            self.pos_emb = relative_position_bias.unsqueeze(0)
            self.relative_bias = self.pos_emb

        input_tensor += self.pos_emb
        return input_tensor


class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self,
                 dim,
                 rank=2,
                 seq_length=4,
                 conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, dim, bias=False))
        else:
            self.cpb_mlp = nn.Sequential(nn.Conv1d(self.rank, 512, 1,bias=True),
                                         nn.ReLU(),
                                         nn.Conv1d(512, dim, 1,bias=False))
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1,seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)
        self.conv = conv

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor):
        seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
        if self.deploy:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            if self.rank == 1:
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype = input_tensor.dtype)
                relative_coords_h -= seq_length//2
                relative_coords_h /= (seq_length//2)
                relative_coords_table = relative_coords_h
                self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))
                self.relative_bias = self.pos_emb
            else:
                seq_length = int(seq_length**0.5)
                relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype = input_tensor.dtype)
                relative_coords_w = torch.arange(0, seq_length, device=input_tensor.device, dtype = input_tensor.dtype)
                relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).contiguous().unsqueeze(0)
                relative_coords_table -= seq_length // 2
                relative_coords_table /= (seq_length // 2)
                if not self.conv:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1,2))
                else:
                    self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
                self.relative_bias = self.pos_emb
        input_tensor = input_tensor + self.pos_emb
        return input_tensor


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x


class Downsample(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.norm = LayerNorm2d(dim)
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):
    """
    Conv block based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()
        """
        Args:
            drop_path: drop path.
            layer_scale: layer scale coefficient.
            kernel_size: kernel size.
        """
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, global_feature=None):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x, global_feature


class WindowAttention(nn.Module):
    """
    Window attention based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 resolution=0,
                 seq_length=0):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            resolution: feature resolution.
            seq_length: sequence length.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv2D(window_size=[resolution, resolution],
                                              pretrained_window_size=[resolution, resolution],
                                              num_heads=num_heads,
                                              seq_length=seq_length)

        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.pos_emb_funct(attn, self.resolution ** 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAT(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1.,
                 window_size=7,
                 last=False,
                 layer_scale=None,
                 ct_size=1,
                 do_propagation=False):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            act_layer: activation function.
            norm_layer: normalization layer.
            sr_ratio: input to window size ratio.
            window_size: window size.
            last: last layer flag.
            layer_scale: layer scale coefficient.
            ct_size: spatial dimension of carrier token local window.
            do_propagation: enable carrier token propagation.
        """
        # positional encoding for windowed attention tokens
        self.pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=window_size**2)
        self.norm1 = norm_layer(dim)
        self.square = True if sr_ratio[0] == sr_ratio[1] else False
        # number of carrier tokens per every window
        self.do_sr_hat = True if ((sr_ratio[0] > 1) or (sr_ratio[1] > 1)) else False
        cr_tokens_per_window = ct_size**2 if self.do_sr_hat else 0

        # total number of carrier tokens
        cr_tokens_total = cr_tokens_per_window*sr_ratio[0]*sr_ratio[1]
        self.cr_window = ct_size

        self.attn = WindowAttention(dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    resolution=window_size,
                                    seq_length=window_size**2 + cr_tokens_per_window)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.window_size = window_size

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

        self.sr_ratio = sr_ratio
        if self.do_sr_hat:
            # if do hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = WindowAttention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, resolution=int(cr_tokens_total**0.5),
                seq_length=cr_tokens_total)

            self.hat_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.hat_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            if self.square:
                self.hat_pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=cr_tokens_total)
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            self.upsampler = nn.Upsample(size=window_size, mode='nearest')

        # keep track for the last block to explicitly add carrier tokens to feature maps
        self.last = last
        self.do_propagation = do_propagation

    def forward(self, x, carrier_tokens):
        B, T, N = x.shape
        ct = carrier_tokens
        x = self.pos_embed(x)

        if self.do_sr_hat:
            # do hierarchical attention via carrier tokens
            # first do attention for carrier tokens
            Bg, Ng, Hg = ct.shape

            # ct are located quite differently
            ct = ct_dewindow(ct, self.cr_window*self.sr_ratio[0], self.cr_window*self.sr_ratio[1], self.cr_window)

            if self.square:
                ct = self.hat_pos_embed(ct)

            # attention plus mlp
            ct = ct + self.hat_drop_path(self.gamma1*self.hat_attn(self.hat_norm1(ct)))
            ct = ct + self.hat_drop_path(self.gamma2*self.hat_mlp(self.hat_norm2(ct)))

            # ct are put back to windows
            ct = ct_window(ct, self.cr_window * self.sr_ratio[0], self.cr_window * self.sr_ratio[1], self.cr_window)

            ct = ct.reshape(x.shape[0], -1, N)
            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)

        # window attention together with carrier tokens
        x = x + self.drop_path(self.gamma3*self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4*self.mlp(self.norm2(x)))

        if self.do_sr_hat:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split([x.shape[1] - self.window_size*self.window_size, self.window_size*self.window_size], dim=1)
            ct = ctr.reshape(Bg, Ng, Hg) # reshape carrier tokens.
            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(B, N, self.cr_window, self.cr_window)
                x = x + self.gamma1 * self.upsampler(ctr_image_space.to(dtype=torch.float32)).flatten(2).transpose(1, 2).to(dtype=x.dtype)
        return x, ct


class TokenInitializer(nn.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """
    def __init__(self,
                 dim,
                 input_resolution,
                 window_size,
                 ct_size=1):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window
        """
        super().__init__()

        output_size = int(ct_size * input_resolution/window_size)
        stride_size = int(input_resolution/output_size)
        kernel_size = input_resolution - (output_size - 1) * stride_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        to_global_feature = nn.Sequential()
        to_global_feature.add_module("pos", self.pos_embed)
        to_global_feature.add_module("pool", nn.AvgPool2d(kernel_size=kernel_size, stride=stride_size))
        self.to_global_feature = to_global_feature
        self.window_size = ct_size

    def forward(self, x):
        x = self.to_global_feature(x)
        B, C, H, W = x.shape
        ct = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        ct = ct.permute(0, 2, 4, 3, 5, 1).reshape(-1, H*W, C)
        return ct

import math
import os

import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ

from .utils import weight_init


class MLPBlock(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, expand * dim),
            nn.ReLU6(),
            nn.Linear(expand * dim, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer"""
    def __init__(self, hidden_dim, condition_dim):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, hidden_dim)
        self.beta = nn.Linear(condition_dim, hidden_dim)
    
    def forward(self, x, condition):
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        return gamma * x + beta


class LatentActHead(nn.Module):
    def __init__(self, act_dim, emb_dim, hidden_dim, expand=4, dropout=0.0):
        super().__init__()
        #TODO : modify 
        # self.proj0 = nn.Linear(2 * emb_dim, hidden_dim)
        # self.proj1 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        # self.proj2 = nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim)
        # self.proj_end = nn.Linear(hidden_dim, act_dim)
        self.proj0 = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.proj1 = nn.Sequential(
            nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.proj2 = nn.Sequential(
            nn.Linear(hidden_dim + 2 * emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.proj_end = nn.Linear(hidden_dim, act_dim)
        
        self.block0 = MLPBlock(hidden_dim, expand, dropout)
        self.block1 = MLPBlock(hidden_dim, expand, dropout)
        self.block2 = MLPBlock(hidden_dim, expand, dropout)

    def forward(self, obs_emb, next_obs_emb):
        x = self.block0(self.proj0(torch.concat([obs_emb, next_obs_emb], dim=-1)))
        x = self.block1(self.proj1(torch.concat([x, obs_emb, next_obs_emb], dim=-1)))
        x = self.block2(self.proj2(torch.concat([x, obs_emb, next_obs_emb], dim=-1)))
        x = self.proj_end(x)
        return x


class LatentObsHead(nn.Module):
    def __init__(self, act_dim, proj_dim, hidden_dim, expand=4, dropout=0.0, camera_embedding_dim=None):
        super().__init__()
        self.use_camera_conditioning = camera_embedding_dim is not None
        
        # 기존 코드 유지
        self.proj0 = nn.Linear(act_dim + proj_dim, hidden_dim)
        self.proj1 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj2 = nn.Linear(act_dim + hidden_dim, hidden_dim)
        self.proj_end = nn.Linear(hidden_dim, proj_dim)

        self.block0 = MLPBlock(hidden_dim, expand, dropout)
        self.block1 = MLPBlock(hidden_dim, expand, dropout)
        self.block2 = MLPBlock(hidden_dim, expand, dropout)
        
        # FiLM layers 추가 (camera conditioning 사용 시)
        if self.use_camera_conditioning:
            self.film0 = FiLMLayer(hidden_dim, camera_embedding_dim)
            self.film1 = FiLMLayer(hidden_dim, camera_embedding_dim)
            self.film2 = FiLMLayer(hidden_dim, camera_embedding_dim)

    def forward(self, x, action, camera_emb=None):
        # Block 0
        x = self.block0(self.proj0(torch.concat([x, action], dim=-1)))
        if self.use_camera_conditioning and camera_emb is not None:
            x = self.film0(x, camera_emb)
        
        # Block 1
        x = self.block1(self.proj1(torch.concat([x, action], dim=-1)))
        if self.use_camera_conditioning and camera_emb is not None:
            x = self.film1(x, camera_emb)
        
        # Block 2
        x = self.block2(self.proj2(torch.concat([x, action], dim=-1)))
        if self.use_camera_conditioning and camera_emb is not None:
            x = self.film2(x, camera_emb)
        
        # Final projection
        x = self.proj_end(x)
        return x


# inspired by:
# 1. https://github.com/schmidtdominik/LAPO/blob/main/lapo/models.py
# 2. https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU6(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2, dropout=0.0, downscale=True):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self._downscale = downscale
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self._downscale else 1,
        )
        # conv downsampling is faster that maxpool, with same perf
        # self.conv = nn.Conv2d(
        #     in_channels=self._input_shape[0],
        #     out_channels=self._out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels, dropout) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.conv(x)
        # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.blocks(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        if self._downscale:
            return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self._out_channels, h, w)


class DecoderBlock(nn.Module):
    def __init__(self, input_shape, out_channels, num_res_blocks=2):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels

        # upsample + conv works fine, just slower than conv-transpose
        # also: upsample does not work well with orthogonal init (why?)!
        # self.conv = nn.Conv2d(
        #     in_channels=self._input_shape[0],
        #     out_channels=self._out_channels,
        #     kernel_size=3,
        #     padding=1,
        # )
        self.conv = nn.ConvTranspose2d(
            in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=2, stride=2
        )
        self.blocks = nn.Sequential(*[ResidualBlock(self._out_channels) for _ in range(num_res_blocks)])

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        x = self.blocks(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, h * 2, w * 2)


class Actor(nn.Module):
    def __init__(
        self,
        shape,
        num_actions,
        encoder_scale=1,
        encoder_channels=(16, 32, 32),
        encoder_num_res_blocks=1,
        dropout=0.0,
    ):
        super().__init__()
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.final_encoder_shape = shape
        self.encoder = nn.Sequential(
            *conv_stack,
            # nn.Flatten(),
        )
        self.actor_mean = nn.Sequential(
            nn.ReLU6(),
            # works either way...
            # nn.Linear(math.prod(shape), num_actions),
            nn.Linear(shape[0], num_actions),
        )
        self.num_actions = num_actions
        self.apply(weight_init)

    def forward(self, obs):
        out = self.encoder(obs)
        out = out.flatten(2).mean(-1)
        act = self.actor_mean(out)
        return act, out


class ActionDecoder(nn.Module):
    def __init__(self, obs_emb_dim, latent_act_dim, true_act_dim, hidden_dim=128):
        super().__init__()
        self.obs_emb_dim = obs_emb_dim
        self.latent_act_dim = latent_act_dim
        self.true_act_dim = true_act_dim

        self.model = nn.Sequential(
            # nn.Linear(latent_act_dim + obs_emb_dim, hidden_dim)
            nn.Linear(latent_act_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(),
            nn.Linear(hidden_dim, true_act_dim),
        )

    def forward(self, obs_emb, latent_act):
        # hidden = torch.concat([obs_emb, latent_act], dim=-1)
        # true_act_pred = self.model(hidden)
        true_act_pred = self.model(latent_act)
        return true_act_pred


# IDM: (s_t, s_t+1) -> a_t
class IDM(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
    ):
        super().__init__()
        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(in_features=math.prod(shape), out_features=latent_act_dim),
            # nn.LayerNorm(latent_act_dim),
        )

    def forward(self, obs, next_obs):
        # [B, C, H, W] -> [B, 2 * C, H, W]
        concat_obs = torch.concat([obs, next_obs], axis=1)
        latent_action = self.encoder(concat_obs)
        return latent_action


# FDM: (s_t, a_t) -> s_t+1
class FDM(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(*conv_stack)
        self.final_encoder_shape = shape

        # decoder
        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels[::-1]:
            conv_seq = DecoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.decoder = nn.Sequential(
            *conv_stack,
            nn.GELU(),
            nn.Conv2d(encoder_channels[0] * encoder_scale, self.inital_shape[0], kernel_size=1),
            nn.Tanh(),
        )
        self.act_proj = nn.Linear(latent_act_dim, math.prod(self.final_encoder_shape))

    def forward(self, obs, latent_action):
        assert obs.ndim == 4, "expect shape [B, C, H, W]"
        obs_emb = self.encoder(obs)
        act_emb = self.act_proj(latent_action).reshape(-1, *self.final_encoder_shape)
        # concat across channels, [B, C * 2, 1, 1]
        emb = torch.concat([obs_emb, act_emb], dim=1)
        next_obs = self.decoder(emb)
        return next_obs


class LAPO(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
    ):
        super().__init__()
        self.idm = IDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
        )
        self.fdm = FDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
        )
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        next_obs_pred = self.fdm(obs, latent_action)
        return next_obs_pred, latent_action

    @torch.no_grad()
    def label(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        return latent_action


# Not used in final experiments, here just for reference.
class IDMFSQ(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim=128,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        fsq_levels=(2, 2),
    ):
        super().__init__()
        assert latent_act_dim % len(fsq_levels) == 0
        self.latent_act_dim = latent_act_dim
        self.fsq_levels = fsq_levels

        shape = (shape[0] * 2, *shape[1:])
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(in_features=math.prod(shape), out_features=latent_act_dim),
            # nn.LayerNorm(latent_act_dim),
        )
        self.quantizer = FSQ(levels=list(fsq_levels))

    def forward(self, obs, next_obs):
        # [B, C, H, W] -> [B, 2 * C, H, W]
        concat_obs = torch.concat([obs, next_obs], axis=1)
        # [B, la_dim]
        latent_action = self.encoder(concat_obs)
        # [B, la_split, la_dim // la_split]
        latent_action = latent_action.reshape(latent_action.shape[0], self.latent_act_dim // len(self.fsq_levels), -1)
        quantized_latent_action, indices = self.quantizer(latent_action)
        quantized_latent_action = quantized_latent_action.reshape(concat_obs.shape[0], -1)
        assert quantized_latent_action.shape[-1] == self.latent_act_dim

        return quantized_latent_action


class LAPOFSQ(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim=128,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        fsq_levels=(2, 2, 4),
    ):
        super().__init__()
        self.idm = IDMFSQ(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
            fsq_levels=fsq_levels,
        )
        self.fdm = FDM(
            shape=shape,
            latent_act_dim=latent_act_dim,
            encoder_scale=encoder_scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
        )
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        next_obs_pred = self.fdm(obs, latent_action)
        return next_obs_pred, latent_action

    @torch.no_grad()
    def label(self, obs, next_obs):
        latent_action = self.idm(obs, next_obs)
        return latent_action


class LAOM(nn.Module):
    def __init__(
        self,
        shape,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        self.act_head = LatentActHead(latent_act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)
        self.obs_head = LatentObsHead(latent_act_dim, math.prod(shape), obs_head_dim, dropout=obs_head_dropout)
        self.final_encoder_shape = shape
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])

        latent_action = self.act_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        latent_next_obs = self.obs_head(obs_emb.flatten(1).detach(), latent_action)

        return latent_next_obs, latent_action, obs_emb.detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        latent_action = self.act_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return latent_action


class LAOMWithLabels(nn.Module):
    def __init__(
        self,
        shape,
        true_act_dim,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
        num_cameras=0,  # 새로 추가
        camera_embedding_dim=64,  # 새로 추가
    ):
        super().__init__()
        self.inital_shape = shape
        
        # Camera conditioning 활성화 조건
        self.use_camera_conditioning = num_cameras > 0
        
        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        self.idm_head = LatentActHead(latent_act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)
        self.true_actions_head = nn.Linear(latent_act_dim, true_act_dim)

        self.fdm_head = LatentObsHead(
            latent_act_dim, 
            math.prod(shape), 
            obs_head_dim, 
            dropout=obs_head_dropout,
            camera_embedding_dim=camera_embedding_dim if self.use_camera_conditioning else None
        )
        
        # Camera embedding layer (조건부 생성)
        if self.use_camera_conditioning:
            self.camera_embedding = nn.Embedding(num_cameras, camera_embedding_dim)
        
        self.final_encoder_shape = shape
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs, predict_true_act=False, future_camera_ids=None):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])

        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        
        # Camera embedding 추출 (조건부)
        camera_emb = None
        if self.use_camera_conditioning and future_camera_ids is not None:
            camera_emb = self.camera_embedding(future_camera_ids)
        
        # FDM with camera conditioning
        latent_next_obs = self.fdm_head(obs_emb.flatten(1).detach(), latent_action, camera_emb)
        # TODO: use norm from encoder here too!

        if predict_true_act:
            true_action = self.true_actions_head(latent_action)
            return latent_next_obs, latent_action, true_action, obs_emb.flatten(1).detach()

        return latent_next_obs, latent_action, obs_emb.flatten(1).detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return latent_action


class IDMLabels(nn.Module):
    def __init__(
        self,
        shape,
        act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        act_head_dim=512,
        act_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            # nn.LayerNorm(math.prod(shape))
        )
        self.idm_head = LatentActHead(act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)

        self.act_dim = act_dim
        self.final_encoder_shape = shape
        self.apply(weight_init)

    def forward(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        pred_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))

        return pred_action, obs_emb.flatten(1).detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        pred_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return pred_action

class MVWithLabels(nn.Module):
    def __init__(
        self,
        shape,
        true_act_dim,
        latent_act_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
    ):
        super().__init__()
        self.inital_shape = shape

        # encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)

        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        self.idm_head = LatentActHead(latent_act_dim, math.prod(shape), act_head_dim, dropout=act_head_dropout)
        self.true_actions_head = nn.Linear(latent_act_dim, true_act_dim)

        self.fdm_head = LatentObsHead(latent_act_dim, math.prod(shape), obs_head_dim, dropout=obs_head_dropout)
        self.final_encoder_shape = shape
        self.latent_act_dim = latent_act_dim
        self.apply(weight_init)

    def forward(self, obs, next_obs, predict_true_act=False):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])

        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        latent_next_obs = self.fdm_head(obs_emb.flatten(1).detach(), latent_action)
        # TODO: use norm from encoder here too!

        if predict_true_act:
            true_action = self.true_actions_head(latent_action)
            return latent_next_obs, latent_action, true_action, obs_emb.flatten(1).detach()

        return latent_next_obs, latent_action, obs_emb.flatten(1).detach()

    @torch.no_grad()
    def label(self, obs, next_obs):
        # for faster forwad + unified batch norm stats, WARN: 2x batch size!
        obs_emb, next_obs_emb = self.encoder(torch.concat([obs, next_obs])).split(obs.shape[0])
        latent_action = self.idm_head(obs_emb.flatten(1), next_obs_emb.flatten(1))
        return latent_action


# ==================== Disentanglement Loss Functions ====================

def compute_infonce_loss(embeddings_list, ids, temperature=0.07, loss_name=""):
    """
    InfoNCE contrastive loss for disentanglement.
    
    Args:
        embeddings_list: List of embedding tensors, each [B, D]
        ids: [B] - IDs for positive pair grouping (same ID = positive)
        temperature: Temperature parameter for InfoNCE
        loss_name: Name for debugging prints.
    
    Returns:
        InfoNCE loss scalar
    """
    # Concat all embeddings: [N, D] where N = len(embeddings_list) * B
    all_embeddings = torch.cat(embeddings_list, dim=0)  # [N, D]
    N = all_embeddings.shape[0]
    
    # Expand ids for all embeddings
    all_ids = ids.repeat(len(embeddings_list))  # [N]
    
    # Compute similarity matrix: [N, N]
    all_embeddings_norm = torch.nn.functional.normalize(all_embeddings, dim=-1)
    similarity_matrix = torch.matmul(all_embeddings_norm, all_embeddings_norm.T) / temperature
    
    # Create positive mask: same id
    positive_mask = (all_ids.unsqueeze(0) == all_ids.unsqueeze(1)).float()  # [N, N]

    # --- DEBUG PRINTS ---
    if "DEBUG_CONTRASTIVE" in os.environ and os.environ["DEBUG_CONTRASTIVE"] == "1":
        print(f"\n--- Contrastive Loss Debug ({loss_name}) ---")
        print(f"Batch size (N): {N}")
        print(f"First 16 IDs used for grouping: {all_ids[:16].cpu().numpy()}")
        
        # Show positive mask for a small part of the batch
        print_size = min(N, 16)
        print(f"Positive Mask (based on IDs, first {print_size}x{print_size}):\n{positive_mask[:print_size, :print_size].cpu().numpy()}")
        
        # Example for the first sample
        anchor_positives_before_self = torch.where(positive_mask[0] == 1)[0]
        print(f"Anchor 0 (ID: {all_ids[0]}) is positive with (including self): {anchor_positives_before_self.cpu().numpy()}")
        
        negatives = torch.where(positive_mask[0] == 0)[0]
        print(f"Anchor 0 is negative with {len(negatives)} samples.")
        print("----------------------------\n")
    # --- END DEBUG PRINTS ---
    
    # Remove self-similarity (diagonal)
    positive_mask = positive_mask * (1 - torch.eye(N, device=positive_mask.device))
    
    # Check if there are any positives
    num_positives = positive_mask.sum(dim=1)
    if num_positives.max() == 0:
        return torch.tensor(0.0, device=all_embeddings.device)
    
    # Compute InfoNCE loss
    exp_sim = torch.exp(similarity_matrix)
    
    # Sum of positive similarities
    pos_sim = (exp_sim * positive_mask).sum(dim=1)
    
    # Sum of all similarities (exclude self)
    all_sim = exp_sim.sum(dim=1) - torch.diag(exp_sim)
    
    # Loss: -log(pos_sim / all_sim)
    # Only compute for samples that have positives
    valid_mask = num_positives > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=all_embeddings.device)
    
    loss = -torch.log((pos_sim[valid_mask] / (all_sim[valid_mask] + 1e-8)) + 1e-8).mean()
    
    return loss


def compute_action_contrastive_loss(z_a_list, instance_ids, temperature=0.07):
    """
    Contrastive loss for action encoder.
    z_a from same instance_id should be similar (positive pairs).
    
    Args:
        z_a_list: List of [B, D] action embeddings from different view pairs
        instance_ids: [B] instance identifiers
        temperature: Temperature for InfoNCE
    
    Returns:
        Action contrastive loss
    """
    return compute_infonce_loss(z_a_list, instance_ids, temperature, loss_name="Action (group by instance_id)")


def compute_view_contrastive_loss(z_v_list, view_pair_ids, instance_ids, temperature=0.07):
    """
    Contrastive loss for view encoder.
    z_v from same view_pair_id (but different instance) should be similar.
    
    Args:
        z_v_list: List of [B, D] view embeddings
        view_pair_ids: [B] view pair identifiers
        instance_ids: [B] instance identifiers (used to exclude same instance)
        temperature: Temperature for InfoNCE
    
    Returns:
        View contrastive loss
    """
    # Use view_pair_ids as the grouping criterion
    # But we need to handle the case where same instance should not be positive
    # For simplicity, use view_pair_ids directly
    return compute_infonce_loss(z_v_list, view_pair_ids, temperature, loss_name="View (group by view_pair_id)")


def compute_zero_regularization(z, eps=1e-6):
    """
    L2 regularization to push embeddings toward zero.
    
    Args:
        z: [B, D] embeddings
        eps: Small constant for numerical stability
    
    Returns:
        L2 norm squared
    """
    return torch.mean(torch.sum(z ** 2, dim=-1))


def compute_reconstruction_loss(pred_emb_list, target_emb_list):
    """
    MSE reconstruction loss between predicted and target embeddings.
    
    Args:
        pred_emb_list: List of predicted embeddings [B, D]
        target_emb_list: List of target embeddings [B, D]
    
    Returns:
        MSE loss
    """
    total_loss = 0.0
    for pred, target in zip(pred_emb_list, target_emb_list):
        total_loss += torch.nn.functional.mse_loss(pred, target)
    return total_loss / len(pred_emb_list)


class DisentangledLAOM(nn.Module):
    """
    Disentangled representation learning model with shared encoder and specialized heads.
    
    Architecture:
    - Shared encoder: obs -> embedding
    - IDM Action: (obs1, obs2) -> z_a (action representation)
    - IDM View: (obs1, obs2) -> z_v (view representation)  
    - FDM: (obs, z_a, z_v) -> obs_pred (reconstruction)
    """
    def __init__(
        self,
        shape,
        latent_act_dim,
        latent_view_dim,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
        obs_head_dim=512,
        obs_head_dropout=0.0,
        separate_fdm_heads=False,
    ):
        super().__init__()
        self.initial_shape = shape
        self.latent_act_dim = latent_act_dim
        self.latent_view_dim = latent_view_dim
        self.separate_fdm_heads = separate_fdm_heads
        
        # Build shared encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)
        
        self.shared_encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        
        # Specialized heads
        emb_dim = math.prod(shape)
        self.idm_action = LatentActHead(latent_act_dim, emb_dim, act_head_dim, dropout=act_head_dropout)
        self.idm_view = LatentActHead(latent_view_dim, emb_dim, act_head_dim, dropout=act_head_dropout)
        
        # FDM heads: either one combined or two separate
        if separate_fdm_heads:
            self.fdm_head_action = LatentObsHead(latent_act_dim, emb_dim, obs_head_dim, dropout=obs_head_dropout)
            self.fdm_head_view = LatentObsHead(latent_view_dim, emb_dim, obs_head_dim, dropout=obs_head_dropout)
        else:
            self.fdm_head = LatentObsHead(latent_act_dim + latent_view_dim, emb_dim, obs_head_dim, dropout=obs_head_dropout)
        
        self.final_encoder_shape = shape
        self.apply(weight_init)
    
    def forward(self, obs1, obs2):
        """
        Forward pass for disentangled representation learning.
        
        Args:
            obs1: First observation [B, C, H, W]
            obs2: Second observation [B, C, H, W]
            
        Returns:
            z_a: Action representation [B, latent_act_dim]
            z_v: View representation [B, latent_view_dim]
            obs_pred: Reconstructed observation [B, emb_dim]
            obs1_emb: First observation embedding [B, emb_dim]
        """
        # Shared encoding (faster forward + unified batch norm stats)
        obs1_emb, obs2_emb = self.shared_encoder(torch.concat([obs1, obs2])).split(obs1.shape[0])
        obs1_emb = obs1_emb.flatten(1)
        obs2_emb = obs2_emb.flatten(1)
        
        # Disentangled predictions
        z_a = self.idm_action(obs1_emb, obs2_emb)  # Action representation
        z_v = self.idm_view(obs1_emb, obs2_emb)    # View representation
        
        # Reconstruction (optional, for reconstruction loss)
        if self.separate_fdm_heads:
            # Use separate heads for action and view reconstruction
            obs_pred_action = self.fdm_head_action(obs1_emb, z_a)
            obs_pred_view = self.fdm_head_view(obs1_emb, z_v)
            obs_pred = torch.concat([obs_pred_action, obs_pred_view], dim=-1)
        else:
            # Use combined head for reconstruction
            z_combined = torch.concat([z_a, z_v], dim=-1)
            obs_pred = self.fdm_head(obs1_emb, z_combined)
        
        return z_a, z_v, obs_pred, obs1_emb.detach()
    
    @torch.no_grad()
    def encode(self, obs):
        """Encode observation to embedding."""
        return self.shared_encoder(obs).flatten(1)
    
    @torch.no_grad()
    def predict_action(self, obs1, obs2):
        """Predict action representation only."""
        obs1_emb, obs2_emb = self.shared_encoder(torch.concat([obs1, obs2])).split(obs1.shape[0])
        return self.idm_action(obs1_emb.flatten(1), obs2_emb.flatten(1))
    
    @torch.no_grad()
    def predict_view(self, obs1, obs2):
        """Predict view representation only."""
        obs1_emb, obs2_emb = self.shared_encoder(torch.concat([obs1, obs2])).split(obs1.shape[0])
        return self.idm_view(obs1_emb.flatten(1), obs2_emb.flatten(1))


class IDMActionOnly(nn.Module):
    """
    Action-only IDM model for BC training.
    
    Architecture:
    - Encoder: obs -> embedding
    - IDM Action: (obs1, obs2) -> z_a (action representation)
    - Action Head: z_a -> true_action (optional, for BC)
    """
    def __init__(
        self,
        shape,
        latent_act_dim,
        true_act_dim=None,
        encoder_scale=1,
        encoder_channels=(16, 32, 64, 128, 256),
        encoder_num_res_blocks=1,
        encoder_dropout=0.0,
        encoder_norm_out=True,
        act_head_dim=512,
        act_head_dropout=0.0,
    ):
        super().__init__()
        self.initial_shape = shape
        self.latent_act_dim = latent_act_dim
        
        # Build encoder
        conv_stack = []
        for out_ch in encoder_channels:
            conv_seq = EncoderBlock(shape, encoder_scale * out_ch, encoder_num_res_blocks, encoder_dropout)
            shape = conv_seq.get_output_shape()
            conv_stack.append(conv_seq)
        
        self.encoder = nn.Sequential(
            *conv_stack,
            nn.Flatten(),
            nn.LayerNorm(math.prod(shape), elementwise_affine=False) if encoder_norm_out else nn.Identity(),
        )
        
        # Action prediction head
        emb_dim = math.prod(shape)
        self.idm_head = LatentActHead(latent_act_dim, emb_dim, act_head_dim, dropout=act_head_dropout)
        
        # Optional: true action prediction head (for BC training)
        if true_act_dim is not None:
            self.action_head = nn.Linear(latent_act_dim, true_act_dim)
        else:
            self.action_head = None
        
        self.final_encoder_shape = shape
        self.apply(weight_init)
    
    def forward(self, obs1, obs2, predict_true_act=False):
        """
        Forward pass for action prediction.
        
        Args:
            obs1: First observation [B, C, H, W]
            obs2: Second observation [B, C, H, W]
            predict_true_act: Whether to predict true action
            
        Returns:
            z_a: Action representation [B, latent_act_dim]
            true_action: True action prediction [B, true_act_dim] (if predict_true_act=True)
            obs1_emb: First observation embedding [B, emb_dim] (if predict_true_act=True)
        """
        # Encoding (faster forward + unified batch norm stats)
        obs1_emb, obs2_emb = self.encoder(torch.concat([obs1, obs2])).split(obs1.shape[0])
        obs1_emb = obs1_emb.flatten(1)
        obs2_emb = obs2_emb.flatten(1)
        
        # Action prediction
        z_a = self.idm_head(obs1_emb, obs2_emb)
        
        if predict_true_act and self.action_head is not None:
            true_action = self.action_head(z_a)
            return z_a, true_action, obs1_emb.detach()
        
        return z_a
    
    @torch.no_grad()
    def encode(self, obs):
        """Encode observation to embedding."""
        return self.encoder(obs).flatten(1)
    
    @torch.no_grad()
    def predict_action(self, obs1, obs2):
        """Predict action representation only."""
        obs1_emb, obs2_emb = self.encoder(torch.concat([obs1, obs2])).split(obs1.shape[0])
        return self.idm_head(obs1_emb.flatten(1), obs2_emb.flatten(1))


def copy_components(source_model, target_model):
    """
    Copy compatible components from DisentangledLAOM to IDMActionOnly.
    
    Args:
        source_model: DisentangledLAOM instance
        target_model: IDMActionOnly instance
        
    Returns:
        target_model: Updated target model with copied weights
    """
    # Copy encoder weights
    target_model.encoder.load_state_dict(source_model.shared_encoder.state_dict())
    
    # Copy IDM action head weights
    target_model.idm_head.load_state_dict(source_model.idm_action.state_dict())
    
    return target_model
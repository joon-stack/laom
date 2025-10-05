import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional
import yaml

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import wandb
from pyrallis import field
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from src.augmentations import Augmenter
from src.nn import Actor, IDMLabels
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    DCSInMemoryDataset,
    DCSLAOMInMemoryDataset,
    create_env_from_df,
    get_grad_norm,
    get_optim_groups,
    normalize_img,
    set_seed,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class IDMConfig:
    total_updates: int = 2500
    batch_size: int = 256
    use_aug: bool = True
    future_obs_offset: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = None
    act_head_dim: int = 512
    act_head_dropout: float = 0.0
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 1
    encoder_deep: bool = True
    encoder_dropout: float = 0.0
    frame_stack: int = 3
    data_path: str = "data/test.hdf5"
    eval_data_path: Optional[str] = None


@dataclass
class BCConfig:
    num_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 2
    encoder_deep: bool = False
    dropout: float = 0.0
    use_aug: bool = True
    frame_stack: int = 3
    data_path: str = "data/test.hdf5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 10
    eval_seed: int = 0


@dataclass
class Config:
    project: str = "laom"
    group: str = "idm"
    name: str = "idm"
    seed: int = 0
    idm_checkpoint_path: Optional[str] = None
    bc_checkpoint_path: Optional[str] = None

    idm: IDMConfig = field(default_factory=IDMConfig)
    bc: BCConfig = field(default_factory=BCConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"


@torch.no_grad()
def evaluate(idm, dataloader, device):
    idm.eval()
    total_samples, total_loss = 0, 0.0

    for batch in dataloader:
        obs, next_obs, _, actions, _, _ = [b.to(device) for b in batch]
        obs = normalize_img(obs.permute((0, 3, 1, 2)))
        next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))

        with torch.autocast(device, dtype=torch.bfloat16):
            pred_action, _ = idm(obs, next_obs)
            eval_loss = F.mse_loss(pred_action, actions, reduction="sum")

        total_loss += eval_loss.item()
        total_samples += obs.shape[0]

    idm.train()
    return total_loss / total_samples


def train_idm(config: IDMConfig, checkpoint_dir: str):
    dataset = DCSLAOMInMemoryDataset(
        config.data_path, max_offset=config.future_obs_offset, frame_stack=config.frame_stack, device=DEVICE
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    num_epochs = config.total_updates // len(dataloader)

    print("INFO: config.total_updates", config.total_updates)
    print("INFO: len(dataloader)", len(dataloader))
    print("INFO: num_epochs", num_epochs)

    if config.eval_data_path is not None:
        eval_dataset = DCSLAOMInMemoryDataset(
            config.eval_data_path, max_offset=1, frame_stack=config.frame_stack, device=DEVICE
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )

    idm = IDMLabels(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        act_dim=dataset.act_dim,
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        encoder_dropout=config.encoder_dropout,
    ).to(DEVICE)

    torchinfo.summary(
        idm,
        input_size=[
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        ],
    )
    optim = torch.optim.Adam(
        params=get_optim_groups(idm, config.weight_decay),
        lr=config.learning_rate,
        fused=True,
    )
    augmenter = Augmenter(dataset.img_hw)

    state_probe = nn.Linear(math.prod(idm.final_encoder_shape), dataset.state_dim).to(DEVICE)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)

    # scheduler setup
    total_updates = len(dataloader) * num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    start_time = time.time()
    total_steps = 0
    total_tokens = 0
    for epoch in trange(num_epochs, desc="Epochs"):
        idm.train()
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            total_tokens += config.batch_size
            total_steps += 1

            obs, _, future_obs, actions, states, _ = [b.to(DEVICE) for b in batch]
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            future_obs = normalize_img(future_obs.permute((0, 3, 1, 2)))

            if config.use_aug:
                obs = augmenter(obs)
                future_obs = augmenter(future_obs)

            # update idm
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action, obs_emb = idm(obs, future_obs)
                loss = F.mse_loss(pred_action, actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(idm.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()

            # evaluation
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_emb.detach())
                state_probe_loss = F.mse_loss(pred_states, states)

            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()

            wandb.log(
                {
                    "idm/mse_loss": loss.item(),
                    "idm/state_probe_loss": state_probe_loss.item(),
                    "idm/throughput": total_tokens / (time.time() - start_time),
                    "idm/learning_rate": scheduler.get_last_lr()[0],
                    # "idm/grad_norm": get_grad_norm(idm).item(),
                    "idm/obs_hidden_norm": torch.norm(obs_emb, p=2, dim=-1).mean().item(),
                    "idm/epoch": epoch,
                    "idm/total_steps": total_steps,
                }
            )

        if config.eval_data_path is not None:
            eval_mse_loss = evaluate(idm, eval_dataloader, device=DEVICE)
            wandb.log({"idm/eval_mse_loss": eval_mse_loss, "idm/epoch": epoch, "idm/total_steps": total_steps})

    # 최종 모델 저장
    save_checkpoint(
        idm,
        optim,
        scheduler,
        num_epochs - 1,
        loss.item(),
        os.path.join(checkpoint_dir, "idm_final.pt"),
        config,
    )

    return idm


@torch.no_grad()
def evaluate_bc(env, actor, num_episodes, seed=0, device="cpu", action_decoder=None):
    returns = []
    for ep in trange(num_episodes, desc="Evaluating", leave=False):
        total_reward = 0.0
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs_ = torch.tensor(obs.copy(), device=device)[None].permute(0, 3, 1, 2)
            obs_ = normalize_img(obs_)
            action, obs_emb = actor(obs_)
            if action_decoder is not None:
                action = action_decoder(action)

            obs, reward, terminated, truncated, info = env.step(action.squeeze().cpu().numpy())
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


def train_bc(lam: IDMLabels, config: BCConfig, checkpoint_dir: str):
    dataset = DCSInMemoryDataset(config.data_path, frame_stack=config.frame_stack, device=DEVICE)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_env = create_env_from_df(
        config.data_path,
        config.dcs_backgrounds_path,
        config.dcs_backgrounds_split,
        frame_stack=config.frame_stack,
    )
    print(eval_env.observation_space)
    print(eval_env.action_space)

    num_actions = dataset.act_dim
    for p in lam.parameters():
        p.requires_grad_(False)
    lam.eval()

    actor = Actor(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        num_actions=num_actions,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        dropout=config.dropout,
    ).to(DEVICE)

    optim = torch.optim.AdamW(params=get_optim_groups(actor, config.weight_decay), lr=config.learning_rate, fused=True)
    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    torchinfo.summary(actor, input_size=(1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw))
    if config.use_aug:
        augmenter = Augmenter(img_resolution=dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        actor.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False):
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, debug_true_actions = [b.to(DEVICE) for b in batch]
            # rescale from 0..255 -> -1..1
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))

            # label with idm latent actions
            target_actions = lam.label(obs, next_obs)

            # augment obs only for bc to make action labels determenistic
            if config.use_aug:
                obs = augmenter(obs)

            # update actor
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_actions, _ = actor(obs)
                loss = F.mse_loss(pred_actions, target_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            wandb.log(
                {
                    "bc/mse_loss": loss.item(),
                    "bc/throughput": total_tokens / (time.time() - start_time),
                    "bc/learning_rate": scheduler.get_last_lr()[0],
                    "bc/epoch": epoch,
                    "bc/total_steps": total_steps,
                }
            )

    # 최종 모델 저장
    save_checkpoint(
        actor,
        optim,
        scheduler,
        config.num_epochs - 1,
        loss.item(),
        os.path.join(checkpoint_dir, "bc_final.pt"),
        config,
    )

    actor.eval()
    eval_returns = evaluate_bc(
        eval_env,
        actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
        action_decoder=None,
    )
    wandb.log(
        {
            "bc/eval_returns_mean": eval_returns.mean(),
            "bc/eval_returns_std": eval_returns.std(),
            "bc/epoch": epoch,
            "bc/total_steps": total_steps,
        }
    )
    return actor


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, config=None):
    """모델 체크포인트를 저장합니다."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }

    # config 정보가 있으면 별도 yaml 파일로 저장
    if config is not None:
        config_filepath = filepath.replace('.pt', '_config.yaml')
        with open(config_filepath, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, allow_unicode=True)
        print(f"Config 저장됨: {config_filepath}")

    torch.save(checkpoint_data, filepath)
    print(f"체크포인트 저장됨: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath):
    """모델 체크포인트를 불러옵니다."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"체크포인트 불러옴: {filepath} (epoch {checkpoint['epoch']})")

        # config yaml 파일이 있으면 불러와서 출력
        config_filepath = filepath.replace('.pt', '_config.yaml')
        if os.path.exists(config_filepath):
            with open(config_filepath, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Config 파일 불러옴: {config_filepath}")
            print("체크포인트에 저장된 config 정보:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            return checkpoint['epoch'], checkpoint['loss'], config

        return checkpoint['epoch'], checkpoint['loss'], None
    return 0, float('inf'), None


@pyrallis.wrap()
def train(config: Config):
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    print(config.bc.eval_episodes)

    set_seed(config.seed)

    # 체크포인트 디렉토리 설정
    base_checkpoint_dir = "/shared/s2/lab01/youngjoonjeong/LAOM/checkpoints"
    if config.idm_checkpoint_path:
        checkpoint_dir = os.path.dirname(config.idm_checkpoint_path)
    elif config.bc_checkpoint_path:
        checkpoint_dir = os.path.dirname(config.bc_checkpoint_path)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = os.path.join(base_checkpoint_dir, timestamp)

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # stage 1: pretraining lapo on unlabeled dataset
    if config.idm_checkpoint_path:
        print("=== Stage 1: IDM Pretraining (skipped, loading from checkpoint) ===")
        # To create model, we need dataset metadata
        dataset = DCSLAOMInMemoryDataset(
            config.idm.data_path,
            max_offset=config.idm.future_obs_offset,
            frame_stack=config.idm.frame_stack,
            device=DEVICE,
        )
        idm = IDMLabels(
            shape=(3 * config.idm.frame_stack, dataset.img_hw, dataset.img_hw),
            act_dim=dataset.act_dim,
            act_head_dim=config.idm.act_head_dim,
            act_head_dropout=config.idm.act_head_dropout,
            encoder_scale=config.idm.encoder_scale,
            encoder_channels=(16, 32, 64, 128, 256) if config.idm.encoder_deep else (16, 32, 32),
            encoder_num_res_blocks=config.idm.encoder_num_res_blocks,
            encoder_dropout=config.idm.encoder_dropout,
        ).to(DEVICE)
        load_checkpoint(idm, None, None, config.idm_checkpoint_path)
    else:
        print("=== Stage 1: IDM Pretraining ===")
        idm = train_idm(config=config.idm, checkpoint_dir=checkpoint_dir)

    # stage 2: pretraining bc on idm labeled actions
    if config.bc_checkpoint_path:
        print("=== Stage 2: BC Pretraining (skipped, loading from checkpoint) ===")
        # We need dataset metadata to create the actor model
        dataset = DCSInMemoryDataset(config.bc.data_path, frame_stack=config.bc.frame_stack, device=DEVICE)
        actor = Actor(
            shape=(3 * config.bc.frame_stack, dataset.img_hw, dataset.img_hw),
            num_actions=dataset.act_dim,
            encoder_scale=config.bc.encoder_scale,
            encoder_channels=(16, 32, 64, 128, 256) if config.bc.encoder_deep else (16, 32, 32),
            encoder_num_res_blocks=config.bc.encoder_num_res_blocks,
            dropout=config.bc.dropout,
        ).to(DEVICE)
        load_checkpoint(actor, None, None, config.bc_checkpoint_path)
    else:
        print("=== Stage 2: BC Pretraining ===")
        actor = train_bc(lam=idm, config=config.bc, checkpoint_dir=checkpoint_dir)

    run.finish()
    return idm, actor


if __name__ == "__main__":
    train()

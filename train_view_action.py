import math
import os
import shutil
import time
import uuid
import pickle
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, List, Any
import yaml

import numpy as np
import pyrallis
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import wandb
from pyrallis import field
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from functools import partial
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw
from scipy.spatial.distance import cdist

from src.augmentations import Augmenter
from src.nn import (
    ActionDecoder, 
    Actor, 
    LAOMWithLabels,
    DisentangledLAOM,
    copy_components,
    compute_action_contrastive_loss,
    compute_view_contrastive_loss,
    compute_zero_regularization,
    compute_reconstruction_loss,
)
import torchvision.models as models
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    DCSViewActionInMemoryDataset,
    DCSViewActionTrueActionsDataset,
    DCSMVInMemoryDataset,
    DCSMVTrueActionsDataset,
    create_env_from_df,
    get_grad_norm,
    get_optim_groups,
    normalize_img,
    set_seed,
    soft_update,
    metric_learning_collate_fn,
    view_action_collate_fn,
    evaluation_collate_fn,
)

# Robomimic imports for BC Transformer
try:
    import sys
    sys.path.append('/home/s2/youngjoonjeong/github/robomimic')
    from robomimic.algo.bc import BC_Transformer
    from robomimic.config import config_factory
    from robomimic.utils.obs_utils import ObsUtils
    from robomimic.utils.file_utils import maybe_dict_from_checkpoint
    ROBOMIMIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Robomimic not available: {e}")
    ROBOMIMIC_AVAILABLE = False

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_dataset_indices(dataset, train_size, val_size, seed, indices_dir="indices", dataset_type="labeled"):
    """
    데이터셋 인덱스를 생성하고 저장합니다.
    지정된 train_size와 val_size만큼만 인덱스를 생성하고 나머지는 버립니다.
    
    Args:
        dataset: 전체 데이터셋
        train_size: train 데이터 크기
        val_size: validation 데이터 크기
        seed: 랜덤 시드
        indices_dir: 인덱스 저장 디렉토리
        dataset_type: "labeled" or "unlabeled" - 인덱스 파일명에 포함
    
    Returns:
        tuple: (train_indices, val_indices)
    """
    os.makedirs(indices_dir, exist_ok=True)
    
    total_size = len(dataset)
    
    # Check if we have enough data
    if train_size + val_size > total_size:
        print(f"Warning: Requested train_size ({train_size}) + val_size ({val_size}) > total_size ({total_size})")
        print(f"Adjusting val_size to {total_size - train_size}")
        val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(total_size, generator=generator).tolist()
    
    # 지정된 개수만큼만 인덱스 분할 (나머지는 버림)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    
    # 인덱스 저장
    indices_info = {
        'dataset_type': dataset_type,
        'total_size': total_size,
        'train_size': train_size,
        'val_size': val_size,
        'seed': seed,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'discarded_size': total_size - train_size - val_size
    }
    
    indices_path = os.path.join(indices_dir, f"indices_{dataset_type}_seed_{seed}_train_{train_size}_val_{val_size}.pkl")
    with open(indices_path, 'wb') as f:
        pickle.dump(indices_info, f)
    
    print(f"{dataset_type.capitalize()} dataset indices saved to: {indices_path}")
    print(f"Total {dataset_type} dataset size: {total_size}")
    print(f"Train: {len(train_indices)} samples")
    print(f"Validation: {len(val_indices)} samples")
    print(f"Discarded: {indices_info['discarded_size']} samples ({indices_info['discarded_size']/total_size:.1%})")
    
    return train_indices, val_indices


def load_dataset_indices(indices_path):
    """
    저장된 데이터셋 인덱스를 불러옵니다.
    
    Args:
        indices_path: 인덱스 파일 경로
    
    Returns:
        tuple: (train_indices, val_indices, indices_info)
    """
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"Indices file not found: {indices_path}")
    
    with open(indices_path, 'rb') as f:
        indices_info = pickle.load(f)
    
    train_indices = indices_info['train_indices']
    val_indices = indices_info['val_indices']
    
    print(f"Dataset indices loaded from: {indices_path}")
    print(f"Train: {len(train_indices)} samples")
    print(f"Validation: {len(val_indices)} samples")
    print(f"Discarded: {indices_info['discarded_size']} samples")
    
    return train_indices, val_indices, indices_info


def get_or_create_traj_splits(hdf5_path, train_traj_num=None, val_traj_num=None, seed=0, splits_dir="indices", dataset_type="unlabeled"):
    """
    HDF5의 data/<demo> 키들(trajectory id)을 기준으로 stage별 train/val 용 demo 리스트를 고정 생성/저장/재사용한다.

    Returns:
        (train_demo_names: List[str], val_demo_names: List[str])  # val은 없으면 []
    """
    os.makedirs(splits_dir, exist_ok=True)
    # 읽기
    with h5py.File(hdf5_path, 'r') as f:
        all_demos = sorted(list(f['data'].keys()))

    total = len(all_demos)
    if dataset_type == "unlabeled":
        if train_traj_num is None and val_traj_num is None:
            # 전부 train으로 사용
            train_traj_num = total
            val_traj_num = 0
        elif train_traj_num is None:
            train_traj_num = max(0, total - (val_traj_num or 0))
        elif val_traj_num is None:
            val_traj_num = max(0, total - train_traj_num)
    elif dataset_type == "labeled":
        if train_traj_num is None and val_traj_num is None:
            train_traj_num = int(0.8 * total)
            val_traj_num = total - train_traj_num
        elif train_traj_num is None:
            train_traj_num = max(0, total - (val_traj_num or 0))
        elif val_traj_num is None:
            val_traj_num = max(0, total - train_traj_num)

    if train_traj_num + val_traj_num > total:
        val_traj_num = max(0, total - train_traj_num)

    filename = f"traj_splits_{dataset_type}_seed_{seed}_train_{train_traj_num}_val_{val_traj_num}.pkl"
    path = os.path.join(splits_dir, filename)

    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded traj splits from: {path}")
        train_demo_names = data['train_demo_names']
        val_demo_names = data.get('val_demo_names', [])
        head_n = min(5, len(train_demo_names))
        if head_n > 0:
            print(f"Train demos head ({head_n}): {train_demo_names[:head_n]}")
        head_nv = min(5, len(val_demo_names))
        if head_nv > 0:
            print(f"Val demos head ({head_nv}): {val_demo_names[:head_nv]}")
        return train_demo_names, val_demo_names

    g = np.random.default_rng(seed)
    perm = g.permutation(total)
    train_idx = perm[:train_traj_num]
    val_idx = perm[train_traj_num:train_traj_num + val_traj_num]
    train_demo_names = [all_demos[i] for i in sorted(train_idx)]
    val_demo_names = [all_demos[i] for i in sorted(val_idx)]

    to_save = {
        'dataset_type': dataset_type,
        'seed': seed,
        'total_traj': total,
        'train_traj_num': train_traj_num,
        'val_traj_num': val_traj_num,
        'train_demo_names': train_demo_names,
        'val_demo_names': val_demo_names,
    }
    with open(path, 'wb') as f:
        pickle.dump(to_save, f)
    print(f"Saved traj splits to: {path}")
    head_n = min(5, len(train_demo_names))
    if head_n > 0:
        print(f"Train demos head ({head_n}): {train_demo_names[:head_n]}")
    head_nv = min(5, len(val_demo_names))
    if head_nv > 0:
        print(f"Val demos head ({head_nv}): {val_demo_names[:head_nv]}")
    return train_demo_names, val_demo_names


def print_dataset_usage_info(dataset, train_indices, val_indices, dataset_type):
    """
    데이터셋 사용 현황을 출력합니다.
    
    Args:
        dataset: 전체 데이터셋
        train_indices: train 인덱스
        val_indices: validation 인덱스 (None 가능)
        dataset_type: "labeled" or "unlabeled"
    """
    total_size = len(dataset)
    train_size = len(train_indices)
    val_size = len(val_indices) if val_indices is not None else 0
    discarded_size = total_size - train_size - val_size
    
    print(f"\n{'='*60}")
    print(f"📊 {dataset_type.upper()} 데이터셋 사용 현황")
    print(f"{'='*60}")
    print(f"전체 데이터셋 크기: {total_size:,} samples")
    print(f"Train 데이터: {train_size:,} samples ({train_size/total_size:.1%})")
    if val_size > 0:
        print(f"Validation 데이터: {val_size:,} samples ({val_size/total_size:.1%})")
    if discarded_size > 0:
        print(f"사용하지 않는 데이터: {discarded_size:,} samples ({discarded_size/total_size:.1%})")
    print(f"{'='*60}\n")


def get_or_create_indices(dataset, config, indices_dir="indices", dataset_type="labeled"):
    """
    인덱스 파일이 있으면 불러오고, 없으면 새로 생성합니다.
    
    Args:
        dataset: 전체 데이터셋
        config: 설정 객체 (train_dataset_size, val_dataset_size, eval_seed 포함)
        indices_dir: 인덱스 저장 디렉토리
        dataset_type: "labeled" or "unlabeled" - 인덱스 파일명에 포함
    
    Returns:
        tuple: (train_indices, val_indices)
    """
    # 인덱스 파일명 생성
    if dataset_type == "unlabeled":
        train_size = config.unlabeled_train_dataset_size if config.unlabeled_train_dataset_size is not None else int(len(dataset))
        val_size = config.unlabeled_val_dataset_size if config.unlabeled_val_dataset_size is not None else len(dataset) - train_size
    else:  # labeled
        train_size = config.train_dataset_size if config.train_dataset_size is not None else int(0.8 * len(dataset))
        val_size = config.val_dataset_size if config.val_dataset_size is not None else len(dataset) - train_size
    
    seed = config.seed
    print("INFO: Using seed: ", seed)
    
    # Check if we have enough data
    total_size = len(dataset)
    if train_size + val_size > total_size:
        print(f"Warning: Requested train_size ({train_size}) + val_size ({val_size}) > total_size ({total_size})")
        print(f"Adjusting val_size to {total_size - train_size}")
        val_size = total_size - train_size
    
    indices_filename = f"indices_{dataset_type}_seed_{seed}_train_{train_size}_val_{val_size}.pkl"
    indices_path = os.path.join(indices_dir, indices_filename)
    
    if os.path.exists(indices_path):
        print(f"Loading existing {dataset_type} indices from: {indices_path}")
        train_indices, val_indices, _ = load_dataset_indices(indices_path)
    else:
        print(f"Creating new {dataset_type} indices and saving to: {indices_path}")
        train_indices, val_indices = create_dataset_indices(
            dataset, train_size, val_size, seed, indices_dir, dataset_type
        )
    
    return train_indices, val_indices


class DirectImageDecoder(nn.Module):
    """Direct image to action decoder using ResNet18 pretrained backbone"""
    def __init__(self, input_shape, true_act_dim, hidden_dim=128):
        super().__init__()
        # ResNet18 pretrained backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final FC layer
        
        # Custom head for action prediction
        self.head = nn.Sequential(
            nn.Linear(512, hidden_dim),  # ResNet18 output: 512
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, true_act_dim)
        )
    
    def forward(self, obs):
        # obs: (B, C, H, W)
        features = self.backbone(obs)  # (B, 512)
        action = self.head(features)   # (B, true_act_dim)
        return action


def save_debug_images(obs_v1, obs_v2, future_obs_v1, future_obs_v2, view_ids_v1, view_ids_v2, instance_ids, offsets, save_dir="debug_images", num_samples=12):
    """디버그용 이미지 저장 함수 - View pair 구조에 맞게 수정"""
    os.makedirs(save_dir, exist_ok=True)

    # uint8을 float로 변환하고 0-1 범위로 정규화
    def normalize_for_display(img_tensor):
        if isinstance(img_tensor, torch.Tensor) and img_tensor.dtype == torch.uint8:
            return img_tensor.float() / 255.0
        return img_tensor

    # 다양한 레이아웃(HWC/CHW)에서 첫 프레임만 HWC(3채널)로 변환
    def first_frame_hwc(img_np: np.ndarray) -> np.ndarray:
        # img_np: numpy array
        if img_np.ndim != 3:
            return img_np
        h, w = img_np.shape[0], img_np.shape[1]
        c_last = img_np.shape[2]
        c_first = img_np.shape[0]
        # Case 1: HWC
        if c_last in (3, 9):
            if c_last == 9:
                return img_np[..., 0:3]
            return img_np
        # Case 2: CHW
        if c_first in (3, 9):
            if c_first == 9:
                return img_np[0:3].transpose(1, 2, 0)
            return img_np.transpose(1, 2, 0)
        # Fallback: assume HWC
        return img_np

    num_samples = min(num_samples, obs_v1.shape[0])

    for i in range(num_samples):
        # 2x2 레이아웃: 상단(obs), 하단(future_obs), 좌측(v1), 우측(v2)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # obs_v1 (top-left)
        obs_v1_img = normalize_for_display(obs_v1[i])
        obs_v1_img = obs_v1_img.detach().cpu().numpy() if isinstance(obs_v1_img, torch.Tensor) else obs_v1_img
        obs_v1_display = first_frame_hwc(obs_v1_img)
        axes[0, 0].imshow(obs_v1_display)
        axes[0, 0].set_title(f"obs_v1 (view_id: {view_ids_v1[i].item()})")
        axes[0, 0].axis('off')

        # obs_v2 (top-right)
        obs_v2_img = normalize_for_display(obs_v2[i])
        obs_v2_img = obs_v2_img.detach().cpu().numpy() if isinstance(obs_v2_img, torch.Tensor) else obs_v2_img
        obs_v2_display = first_frame_hwc(obs_v2_img)
        axes[0, 1].imshow(obs_v2_display)
        axes[0, 1].set_title(f"obs_v2 (view_id: {view_ids_v2[i].item()})")
        axes[0, 1].axis('off')

        # future_obs_v1 (bottom-left)
        fut_v1_img = normalize_for_display(future_obs_v1[i])
        fut_v1_img = fut_v1_img.detach().cpu().numpy() if isinstance(fut_v1_img, torch.Tensor) else fut_v1_img
        fut_v1_display = first_frame_hwc(fut_v1_img)
        axes[1, 0].imshow(fut_v1_display)
        axes[1, 0].set_title(f"future_obs_v1 (offset: {offsets[i].item()})")
        axes[1, 0].axis('off')

        # future_obs_v2 (bottom-right)
        fut_v2_img = normalize_for_display(future_obs_v2[i])
        fut_v2_img = fut_v2_img.detach().cpu().numpy() if isinstance(fut_v2_img, torch.Tensor) else fut_v2_img
        fut_v2_display = first_frame_hwc(fut_v2_img)
        axes[1, 1].imshow(fut_v2_display)
        axes[1, 1].set_title(f"future_obs_v2 (offset: {offsets[i].item()})")
        axes[1, 1].axis('off')

        # 전체 제목
        fig.suptitle(f"Sample {i} - instance_id: {instance_ids[i].item()}", fontsize=14)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                save_dir,
                f"sample_{i:02d}_inst_{instance_ids[i].item()}_v1_{view_ids_v1[i].item()}_v2_{view_ids_v2[i].item()}.png",
            ),
            dpi=100,
            bbox_inches='tight',
        )
        plt.close()

    print(f"Debug images saved to {save_dir}/")


@dataclass
class LAOMConfig:
    device: str = "cuda"
    num_epochs: int = 100
    batch_size: int = 32
    labeled_batch_size: int = 256
    camera_num: int = 8
    view_keys: List[Any] = field(default_factory=list)
    eval_view_keys: List[Any] = field(default_factory=list)
    views_per_instance: int = 4  # K: views to sample per instance
    contrastive_loss_coef: float = 1.0  # 통일된 contrastive loss 계수
    triplet_warmup_epochs: int = 0
    base_margin: float = 0.5
    labeled_loss_coef: float = 0.05
    cosine_loss: bool = False
    use_aug: bool = False
    future_obs_offset: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = 1.0
    latent_action_dim: int = 256
    latent_view_dim: int = 256
    act_head_dim: int = 512
    act_head_dropout: float = 0.0
    obs_head_dim: int = 512
    obs_head_dropout: float = 0.0
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 1
    encoder_dropout: float = 0.0
    encoder_norm_out: bool = True
    encoder_deep: bool = True
    target_tau: float = 0.01
    target_update_every: int = 1
    frame_stack: int = 3
    data_path: str = "data/test.hdf5"
    eval_data_path: Optional[str] = None
    labeled_data_path: str = "data/labeled_test.hdf5"
    seed: int = 0
    num_workers: int = 4
    normalize_triplet: bool = True
    loss_type: str = "infonce"  # "triplet" or "infonce"
    infonce_temperature: float = 0.1
    # --- False Negative Filtering ---
    filtering_type: str = "none"  # "none", "dtw", or "l2"
    dtw_threshold: float = 5.0
    l2_threshold: float = 5.0
    fixed_batch_offset: bool = True # Use fixed offset for all samples in a batch for DTW
    
    # View sampling configuration
    mixed_view_sampling: bool = False  # Sample different views per positive sample
    positive_samples_per_instance: int = 4  # Number of positive samples per instance when using mixed view sampling
    resize_size: int = 64  # Image resize size
    
    # --- Loss Coefficients ---
    view_loss_coef: float = 1.0
    zero_act_coef: float = 1.0
    zero_view_coef: float = 1.0
    recon_coef: float = 1.0
    
    # --- Reconstruction Loss Options ---
    exclude_future_v2_from_recon: bool = False
    separate_fdm_heads: bool = False  # If True, use separate fdm_head for action and view

    train_dataset_size: Optional[int] = None
    val_dataset_size: Optional[int] = None
    unlabeled_train_dataset_size: Optional[int] = None
    unlabeled_val_dataset_size: Optional[int] = None

    # Trajectory-level splits (number of demos)
    train_traj_num: Optional[int] = None
    val_traj_num: Optional[int] = None
    unlabeled_train_traj_num: Optional[int] = None
    unlabeled_val_traj_num: Optional[int] = None

    def __post_init__(self):
        if not self.view_keys:
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in range(self.camera_num)]
        elif self.view_keys and isinstance(self.view_keys[0], int):
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in self.view_keys]
        
        if not self.eval_view_keys:
            self.eval_view_keys = self.view_keys
        elif self.eval_view_keys and isinstance(self.eval_view_keys[0], int):
            self.eval_view_keys = [f"view_{i:02d}/agentview_image" for i in self.eval_view_keys]


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
    seed: int = 0  # BC training용 seed
    # Multi-view related settings
    camera_num: int = 20
    view_keys: List[Any] = field(default_factory=list)
    views_per_instance: int = 1  # K: views to sample per instance
    fixed_offset: int = 1  # Fixed offset for all samples
    # Dataset split settings
    train_dataset_size: Optional[int] = None  # If None, use 80:20 split. If set, use this size for train
    val_dataset_size: Optional[int] = None    # If None, use remaining data for validation. If set, use this size for validation (rest will be discarded)
    # Unlabeled dataset size (for LAPO stage)
    unlabeled_train_dataset_size: Optional[int] = None
    unlabeled_val_dataset_size: Optional[int] = None

    # Trajectory-level splits (number of demos)
    train_traj_num: Optional[int] = None
    val_traj_num: Optional[int] = None
    
    # Image resize
    resize_size: int = 64
    
    # View sampling configuration
    mixed_view_sampling: bool = False
    positive_samples_per_instance: int = 4

    def __post_init__(self):
        if not self.view_keys:
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in range(self.camera_num)]
        elif self.view_keys and isinstance(self.view_keys[0], int):
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in self.view_keys]


@dataclass
class DecoderConfig:
    total_updates: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    hidden_dim: int = 128
    use_aug: bool = True
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 10
    eval_seed: int = 0
    # Action decoder type selection
    action_decoder_type: str = "simple"  # "simple" or "bc_transformer"
    bc_transformer_config_path: Optional[str] = None
    # Multi-view related settings (inherited from BCConfig)
    camera_num: int = 20
    view_keys: List[Any] = field(default_factory=list)
    views_per_instance: int = 1
    frame_stack: int = 3
    fixed_offset: int = 1  # Fixed offset for all samples
    # Dataset split settings
    train_dataset_size: Optional[int] = None  # If None, use 80:20 split. If set, use this size for train
    val_dataset_size: Optional[int] = None    # If None, use remaining data for validation. If set, use this size for validation (rest will be discarded)
    # Unlabeled dataset size (for LAPO stage)
    unlabeled_train_dataset_size: Optional[int] = None
    unlabeled_val_dataset_size: Optional[int] = None

    # Trajectory-level splits (number of demos)
    train_traj_num: Optional[int] = None
    val_traj_num: Optional[int] = None
    
    # Image resize
    resize_size: int = 64
    
    # View sampling configuration
    mixed_view_sampling: bool = False
    positive_samples_per_instance: int = 4

    def __post_init__(self):
        if not self.view_keys:
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in range(self.camera_num)]
        elif self.view_keys and isinstance(self.view_keys[0], int):
            self.view_keys = [f"view_{i:02d}/agentview_image" for i in self.view_keys]


@dataclass
class Config:
    project: str = "view-action-disentagle"
    group: str = "mv-labels"
    name: str = "mv-labels"
    run_name: Optional[str] = None  # Custom run name for checkpoint dir and wandb
    seed: int = 0
    debug_dataloader: bool = False  # Add this line for debugging
    lapo_checkpoint_path: Optional[str] = None
    bc_checkpoint_path: Optional[str] = None

    lapo: LAOMConfig = field(default_factory=LAOMConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"
        # coupling labeled dataset for laom pretraining and action decoder finetuning
        self.decoder.data_path = self.lapo.labeled_data_path
        self.lapo.seed = self.seed
        self.decoder.seed = self.seed
        # BCConfig의 seed를 전체 Config의 seed와 동일하게 설정
        self.bc.seed = self.seed
        # Propagate dataset split settings from decoder to bc
        self.bc.train_dataset_size = self.decoder.train_dataset_size
        self.bc.val_dataset_size = self.decoder.val_dataset_size
        self.bc.eval_seed = self.decoder.eval_seed
        # Propagate unlabeled dataset settings
        self.bc.unlabeled_train_dataset_size = self.decoder.unlabeled_train_dataset_size
        self.bc.unlabeled_val_dataset_size = self.decoder.unlabeled_val_dataset_size


@torch.no_grad()
def evaluate(lam, act_probe, dataloader, config, device):
    lam.eval()
    act_probe.eval()
    
    total_samples = 0
    total_losses = {
        "eval_probe_action_mse_loss": 0.0,
        "eval_action_contrastive_loss": 0.0,
    }

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        if batch is None:
            continue
        
        obs = batch["obs"].to(device)
        future_obs = batch["future_obs"].to(device)
        actions = batch["actions"].to(device)
        instance_ids = batch["instance_ids"].to(device)

        obs = normalize_img(obs.permute((0, 3, 1, 2)))
        future_obs = normalize_img(future_obs.permute((0, 3, 1, 2)))

        with torch.autocast(device, dtype=torch.bfloat16):
            # 1. Probe MSE
            z_a, _, _, _ = lam(obs, future_obs)
            pred_action = act_probe(z_a)
            probe_mse = F.mse_loss(pred_action, actions)
            
            # 2. Action Contrastive Loss
            con_act_loss = compute_action_contrastive_loss(
                [z_a],
                instance_ids,
                temperature=config.infonce_temperature
            )

        total_losses["eval_probe_action_mse_loss"] += probe_mse.item() * obs.shape[0]
        total_losses["eval_action_contrastive_loss"] += con_act_loss.item() * obs.shape[0]
        total_samples += obs.shape[0]

    lam.train()
    act_probe.train()
    
    if total_samples == 0:
        return {k: 0.0 for k in total_losses}
        
    avg_losses = {k: v / total_samples for k, v in total_losses.items()}
    
    return avg_losses


def train_mv(config: LAOMConfig, checkpoint_dir: str, config_path: str = None):
    set_seed(config.seed)
    DEVICE = config.device

    # --- Print view key configuration for debugging ---
    print("--- View Key Configuration ---")
    print(f"Training view_keys: {config.view_keys}")
    print(f"Evaluation view_keys: {config.eval_view_keys}")
    print("-----------------------------")
    
    # --- Print view sampling configuration ---
    print("--- View Sampling Configuration ---")
    print(f"Mixed view sampling: {config.mixed_view_sampling}")
    if config.mixed_view_sampling:
        print(f"Views per instance (K): {config.views_per_instance}")
        print(f"Positive samples per instance: {config.positive_samples_per_instance}")
        print(f"Available views: {config.view_keys}")
    print("-----------------------------")
    
    # --- Print loss configuration ---
    print("--- Loss Configuration ---")
    print(f"Loss type: {config.loss_type}")
    print(f"Contrastive loss coefficient: {config.contrastive_loss_coef}")
    if config.loss_type == "triplet":
        print(f"Base margin: {config.base_margin}")
        print(f"Normalize triplet: {config.normalize_triplet}")
    elif config.loss_type == "infonce":
        print(f"InfoNCE temperature: {config.infonce_temperature}")
        print(f"Normalize triplet: {config.normalize_triplet}")
    print("-----------------------------")

    # --- Print Filtering Configuration ---
    print("--- False Negative Filtering Configuration ---")
    print(f"Filtering type: {config.filtering_type}")
    if config.filtering_type == "dtw":
        print(f"DTW threshold: {config.dtw_threshold}")
        print(f"Use fixed batch offset: {config.fixed_batch_offset}")
    elif config.filtering_type == "l2":
        print(f"L2 threshold: {config.l2_threshold}")
        print(f"Use fixed batch offset: {config.fixed_batch_offset}")
    print("------------------------------------------")

    # Stage-1: unlabeled traj splits
    unl_train_demos, _ = get_or_create_traj_splits(
        hdf5_path=config.data_path,
        train_traj_num=config.unlabeled_train_traj_num,
        val_traj_num=config.unlabeled_val_traj_num,
        seed=config.seed,
        splits_dir="indices",
        dataset_type="unlabeled",
    )
    dataset = DCSViewActionInMemoryDataset(
        config.data_path,
        max_offset=config.future_obs_offset,
        frame_stack=config.frame_stack,
        device="cpu",
        camera_num=config.camera_num,
        view_keys=config.view_keys,
        resize_wh=(config.resize_size, config.resize_size),
        selected_demo_names=unl_train_demos,
    )

    # traj 기반 사용: 인덱스 분할 제거
    train_dataset = dataset

    collate_fn = partial(view_action_collate_fn, K=config.views_per_instance)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create labeled dataset with index-based sampling
    # Stage-1 labeled splits for supervised head
    lab_train_demos, lab_val_demos = get_or_create_traj_splits(
        hdf5_path=config.labeled_data_path,
        train_traj_num=config.train_traj_num,
        val_traj_num=config.val_traj_num,
        seed=config.seed,
        splits_dir="indices",
        dataset_type="labeled",
    )

    labeled_dataset = DCSMVTrueActionsDataset(
        config.labeled_data_path,
        max_offset=config.future_obs_offset,
        frame_stack=config.frame_stack,
        device="cpu",
        camera_num=config.camera_num,
        view_keys=config.view_keys,
        resize_wh=(config.resize_size, config.resize_size),
        selected_demo_names=lab_train_demos,
        mixed_view_sampling=config.mixed_view_sampling,
        positive_samples_per_instance=config.positive_samples_per_instance,
    )
    
    # Labeled dataloader (traj 기반) - uses old structure with metric_learning_collate_fn
    labeled_collate_fn = partial(metric_learning_collate_fn, K=config.views_per_instance, mixed_view_sampling=config.mixed_view_sampling)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=config.labeled_batch_size, collate_fn=labeled_collate_fn)

    if config.eval_data_path is not None:
        eval_dataset = DCSMVTrueActionsDataset(
            config.eval_data_path,
            max_offset=config.future_obs_offset,
            frame_stack=config.frame_stack,
            device="cpu",
            camera_num=config.camera_num,
            view_keys=config.eval_view_keys,
            resize_wh=(config.resize_size, config.resize_size),
            selected_demo_names=lab_val_demos if lab_val_demos else None,
            mixed_view_sampling=config.mixed_view_sampling,
            positive_samples_per_instance=config.positive_samples_per_instance,
        )
        eval_collate_fn = partial(metric_learning_collate_fn, K=config.views_per_instance, mixed_view_sampling=config.mixed_view_sampling)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=eval_collate_fn,
            num_workers=config.num_workers,
        )



    # ==================== Disentangled LAOM Model ====================
    # Single model with shared encoder and specialized heads
    disentangled_laom = DisentangledLAOM(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        latent_act_dim=config.latent_action_dim,  # z_a dimension
        latent_view_dim=config.latent_view_dim,  # z_v dimension
        act_head_dim=config.act_head_dim,
        act_head_dropout=config.act_head_dropout,
        obs_head_dim=config.obs_head_dim,
        obs_head_dropout=config.obs_head_dropout,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        encoder_dropout=config.encoder_dropout,
        encoder_norm_out=config.encoder_norm_out,
        separate_fdm_heads=config.separate_fdm_heads,
    ).to(DEVICE)

    # Target network (EMA)
    target_disentangled_laom = deepcopy(disentangled_laom)
    for p in target_disentangled_laom.parameters():
        p.requires_grad_(False)
    
    # Keep old variable names for compatibility with existing code
    laom_action = disentangled_laom
    laom_view = disentangled_laom
    laom_shared = disentangled_laom
    target_laom_action = target_disentangled_laom
    target_laom_view = target_disentangled_laom
    target_laom_shared = target_disentangled_laom
    lapo = disentangled_laom
    target_lapo = target_disentangled_laom

    torchinfo.summary(
        lapo,
        input_size=[
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        ],
    )
    
    # Optimizer for single disentangled model
    optim = torch.optim.Adam(
        params=get_optim_groups(disentangled_laom, config.weight_decay),
        lr=config.learning_rate,
        fused=True,
    )
    augmenter = Augmenter(dataset.img_hw)

    state_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.state_dim).to(DEVICE)
    state_probe_optim = torch.optim.Adam(state_probe.parameters(), lr=config.learning_rate)

    act_linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    act_probe_optim = torch.optim.Adam(act_linear_probe.parameters(), lr=config.learning_rate)

    print("Final encoder shape:", math.prod(lapo.final_encoder_shape))
    state_act_linear_probe = nn.Linear(math.prod(lapo.final_encoder_shape), dataset.act_dim).to(DEVICE)
    state_act_probe_optim = torch.optim.Adam(state_act_linear_probe.parameters(), lr=config.learning_rate)

    # scheduler setup
    total_updates = len(train_dataloader) * config.num_epochs
    warmup_updates = len(train_dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    start_time = time.time()
    total_iterations = 0
    total_tokens = 0
    step_counter = 0  # Add step counter for wandb logging

    labeled_dataloader_iter = iter(labeled_dataloader)
    base_contrastive_loss_coef = config.contrastive_loss_coef
    for epoch in trange(config.num_epochs, desc="Epochs"):
        # Contrastive loss coefficient warm-up
        if epoch < config.triplet_warmup_epochs:
            current_contrastive_coef = 0.0
        else:
            current_contrastive_coef = base_contrastive_loss_coef

        disentangled_laom.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False):
            if i == 1:
                break
            if batch is None: continue
            total_tokens += config.batch_size * config.views_per_instance
            total_iterations += 1

            # action_sequences_mask is added
            obs_v1, obs_v2, future_obs_v1, future_obs_v2, view_pair_ids, instance_ids, offsets, actions, action_sequences, states, action_sequences_mask = [
                batch[k].to(DEVICE) for k in ["obs_v1", "obs_v2", "future_obs_v1", "future_obs_v2", "view_pair_ids", "instance_ids", "offsets", "actions", "action_sequences", "states", "action_sequences_mask"]
            ]
            if i == 0:                                                                                             
                print(f"Unique instance IDs: {torch.unique(instance_ids).cpu().numpy()}")                          
                print(f"Number of unique instances: {len(torch.unique(instance_ids))}")                            
                print(f"Unique view pair IDs: {torch.unique(view_pair_ids).cpu().numpy()}")                        
                print(f"Number of unique view pairs: {len(torch.unique(view_pair_ids))}")
            # Recover view_ids_v1 and view_ids_v2 from view_pair_ids for debugging
            view_ids_v1 = view_pair_ids // 10000
            view_ids_v2 = view_pair_ids % 10000

            # Debug: Print offsets for first batch to check if fixed_batch_offset=False works
            if epoch == 0 and i == 0:
                print(f"\n=== Debug: First batch offsets (fixed_batch_offset={config.fixed_batch_offset}) ===")
                print(f"Batch size: {offsets.shape[0]}")
                print(f"Offsets: {offsets.cpu().numpy()}")
                print(f"Offset range: {offsets.min().item()} - {offsets.max().item()}")
                print(f"Unique offsets: {torch.unique(offsets).cpu().numpy()}")
                print("=" * 60)
                
                # Debug: Check view pair sampling behavior
                print(f"\n=== Debug: View Pair Sampling Check ===")
                print(f"Config positive_samples_per_instance: {config.positive_samples_per_instance}")
                print(f"Config views_per_instance (K): {config.views_per_instance}")
                print(f"Actual batch size: {obs_v1.shape[0]}")
                print(f"Instance IDs: {instance_ids.cpu().numpy()}")
                print(f"View Pair IDs: {view_pair_ids.cpu().numpy()}")
                print(f"View IDs v1: {view_ids_v1.cpu().numpy()}")
                print(f"View IDs v2: {view_ids_v2.cpu().numpy()}")
                # print(f"Unique instance IDs: {torch.unique(instance_ids).cpu().numpy()}")
                # print(f"Number of unique instances: {len(torch.unique(instance_ids))}")
                # print(f"Unique view pair IDs: {torch.unique(view_pair_ids).cpu().numpy()}")
                # print(f"Number of unique view pairs: {len(torch.unique(view_pair_ids))}")
                print("=" * 60)
                
                # Save debug images from actual training dataloader
                print("\n=== Saving Debug Images from Training Dataloader (Unlabeled) ===")
                save_debug_images(
                    obs_v1, 
                    obs_v2, 
                    future_obs_v1,
                    future_obs_v2,
                    view_ids_v1,
                    view_ids_v2,
                    instance_ids,
                    offsets,
                    save_dir=os.path.join(checkpoint_dir, "debug_images_train_unlabeled"),
                    num_samples=min(24, obs_v1.shape[0])
                )

            # View pair data is now shaped (K*P, H, W, C) from collate_fn.
            # Permute to (K*P, C, H, W) for conv networks.
            obs_v1 = obs_v1.permute((0, 3, 1, 2))
            obs_v2 = obs_v2.permute((0, 3, 1, 2))
            future_obs_v1 = future_obs_v1.permute((0, 3, 1, 2))
            future_obs_v2 = future_obs_v2.permute((0, 3, 1, 2))
            
            # Normalization is now applied to the already flattened batch
            obs_v1 = normalize_img(obs_v1)
            obs_v2 = normalize_img(obs_v2)
            future_obs_v1 = normalize_img(future_obs_v1)
            future_obs_v2 = normalize_img(future_obs_v2)

            if config.use_aug:
                obs_v1_to_use = augmenter(obs_v1)
                obs_v2_to_use = augmenter(obs_v2)
                future_obs_v1_to_use = augmenter(future_obs_v1)
                future_obs_v2_to_use = augmenter(future_obs_v2)
            else:
                obs_v1_to_use = obs_v1
                obs_v2_to_use = obs_v2
                future_obs_v1_to_use = future_obs_v1
                future_obs_v2_to_use = future_obs_v2

            # ==================== Disentanglement Training ====================
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                # ========== 1. Extract latent vectors for entire batch (batch-wise) ==========
                batch_size = obs_v1_to_use.shape[0]

                # IDM_action: 4 types of time-changing pairs
                z_a_v1, _, _, _ = disentangled_laom(obs_v1_to_use, future_obs_v1_to_use)
                z_a_v2, _, _, _ = disentangled_laom(obs_v2_to_use, future_obs_v2_to_use)
                z_a_cross1, _, _, _ = disentangled_laom(obs_v1_to_use, future_obs_v2_to_use)
                z_a_cross2, _, _, _ = disentangled_laom(obs_v2_to_use, future_obs_v1_to_use)
                
                # Stack all z_a and corresponding instance_ids
                all_z_a = torch.cat([z_a_v1, z_a_v2, z_a_cross1, z_a_cross2], dim=0)
                all_instance_ids_for_act = instance_ids.repeat(4)

                # IDM_view: 4 types of view-changing pairs
                _, z_v_t1, _, _ = disentangled_laom(obs_v1_to_use, obs_v2_to_use)
                _, z_v_t2, _, _ = disentangled_laom(future_obs_v1_to_use, future_obs_v2_to_use)
                _, z_v_cross1, _, _ = disentangled_laom(obs_v1_to_use, future_obs_v2_to_use)
                _, z_v_cross2, _, _ = disentangled_laom(obs_v2_to_use, future_obs_v1_to_use)

                # Stack all z_v and corresponding view_pair_ids
                all_z_v = torch.cat([z_v_t1, z_v_t2, z_v_cross1, z_v_cross2], dim=0)
                all_view_pair_ids_for_view = view_pair_ids.repeat(4)

                # ========== 2. Zero Regularization Losses ==========
                # When no time change, z_a should be zero
                z_a_zero, _, _, _ = disentangled_laom(obs_v1_to_use, obs_v2_to_use)  # same time, different view
                loss_zero_act = compute_zero_regularization(z_a_zero)
                
                # When no view change, z_v should be zero
                _, z_v_zero, _, _ = disentangled_laom(obs_v1_to_use, future_obs_v1_to_use)  # same view, different time
                loss_zero_view = compute_zero_regularization(z_v_zero)
                
                # ========== 3. Contrastive Losses (Batch-wise) ==========
                # Action contrastive: same instance_id across batch = positive
                loss_con_act = compute_action_contrastive_loss(
                    [all_z_a],  # All z_a from entire batch
                    all_instance_ids_for_act,
                    temperature=config.infonce_temperature if hasattr(config, 'infonce_temperature') else 0.07
                )
                
                # View contrastive: same view_pair_id across batch = positive
                loss_con_view = compute_view_contrastive_loss(
                    [all_z_v],  # All z_v from entire batch
                    all_view_pair_ids_for_view,
                    all_instance_ids_for_act,  # Not used in current implementation
                    temperature=config.infonce_temperature if hasattr(config, 'infonce_temperature') else 0.07
                )
                
                # ========== 4. Reconstruction Loss (FDM) ==========
                # Get target embeddings for the whole batch
                with torch.no_grad():
                    target_future_v1_emb = target_disentangled_laom.encode(future_obs_v1)
                    target_obs_v2_emb = target_disentangled_laom.encode(obs_v2)
                    target_future_v2_emb = target_disentangled_laom.encode(future_obs_v2)
                
                # Get source embedding for the whole batch
                obs_v1_emb = disentangled_laom.encode(obs_v1_to_use)
                
                # Reconstruction based on separate_fdm_heads setting
                if config.separate_fdm_heads:
                    # Use separate heads for action and view reconstruction
                    pred_future_v1_emb = disentangled_laom.fdm_head_action(obs_v1_emb, z_a_v1)
                    pred_obs_v2_emb = disentangled_laom.fdm_head_view(obs_v1_emb, z_v_t1)
                else:
                    # Use single head with zeros for the other latent
                    # Reconstruct with z_a only (action only)
                    z_combined_act_only = torch.cat([z_a_v1, torch.zeros_like(z_v_t1)], dim=-1)
                    pred_future_v1_emb = disentangled_laom.fdm_head(obs_v1_emb, z_combined_act_only)
                    
                    # Reconstruct with z_v only (view only)
                    z_combined_view_only = torch.cat([torch.zeros_like(z_a_v1), z_v_t1], dim=-1)
                    pred_obs_v2_emb = disentangled_laom.fdm_head(obs_v1_emb, z_combined_view_only)
                
                # Reconstruct with both z_a and z_v
                if config.separate_fdm_heads:
                    # For separate heads, we need to handle the combined case
                    # We can either use one of the heads or create a combined prediction
                    # For now, let's use the action head as the primary reconstruction
                    pred_future_v2_emb = disentangled_laom.fdm_head_view(disentangled_laom.fdm_head_action(obs_v1_emb, z_a_v1), z_v_t1)
                else:
                    z_combined_both = torch.cat([z_a_v1, z_v_t1], dim=-1)
                    pred_future_v2_emb = disentangled_laom.fdm_head(obs_v1_emb, z_combined_both)
                
                # Reconstruction loss for the whole batch
                exclude_v2 = getattr(config, 'exclude_future_v2_from_recon', False)
                if exclude_v2:
                    loss_recon = compute_reconstruction_loss(
                        [pred_future_v1_emb, pred_obs_v2_emb],
                        [target_future_v1_emb, target_obs_v2_emb]
                    )
                    # just for validation
                    loss_recon_both = compute_reconstruction_loss(
                        [pred_future_v2_emb],
                        [target_future_v2_emb]
                    )
                else:
                    loss_recon = compute_reconstruction_loss(
                        [pred_future_v1_emb, pred_obs_v2_emb, pred_future_v2_emb],
                        [target_future_v1_emb, target_obs_v2_emb, target_future_v2_emb]
                    )
                
                # ========== 5. Total Loss ==========
                alpha = getattr(config, 'view_loss_coef', 1.0)
                beta = getattr(config, 'zero_act_coef', 1.0)
                gamma = getattr(config, 'zero_view_coef', 1.0)
                delta = getattr(config, 'recon_coef', 1.0)
                
                loss0 = (
                    1.0 * (loss_con_act + alpha * loss_con_view) + 
                    beta * loss_zero_act + 
                    gamma * loss_zero_view + 
                    delta * loss_recon
                )

                # print("INFO: loss0: ", loss0.item())
                # print("INFO: loss_con_act: ", loss_con_act.item())
                # print("INFO: alpha * loss_con_view: ", alpha * loss_con_view.item())
                # print("INFO: beta * loss_zero_act: ", beta * loss_zero_act.item())
                # print("INFO: gamma * loss_zero_view: ", gamma * loss_zero_view.item())
                # print("INFO: delta * loss_recon: ", delta * loss_recon.item())

            #TODO: disabled auxilary action prediction task
            # # loss with true actions
            # labeled_batch = next(labeled_dataloader_iter)
            # if labeled_batch is None: continue
            # # Enable verification after first successful fetch to confirm correctness
            # # Debug: after first labeled batch, print sampling stats once
            # if i == 0 and hasattr(labeled_dataloader.dataset, 'get_sampling_stats'):
            #     print(f"Labeled dataset sampling stats (after first batch): {labeled_dataloader.dataset.get_sampling_stats(reset=True)}")

            # label_obs, label_next_obs, label_future_obs, label_actions = [
            #     labeled_batch[k].to(DEVICE) for k in ["obs", "next_obs", "future_obs", "actions"]
            # ]
            

            # label_obs = label_obs.permute((0, 3, 1, 2))
            # label_next_obs = label_next_obs.permute((0, 3, 1, 2))
            # label_future_obs = label_future_obs.permute((0, 3, 1, 2))

            # label_obs = normalize_img(label_obs)
            # label_next_obs = normalize_img(label_next_obs)
            # label_future_obs = normalize_img(label_future_obs)

            # if config.use_aug:
            #     label_obs_to_use = augmenter(label_obs)
            #     label_future_obs_to_use = augmenter(label_future_obs)
            # else:
            #     label_obs_to_use = label_obs
            #     label_future_obs_to_use = label_future_obs

            # # update lapo
            # with torch.autocast(DEVICE, dtype=torch.bfloat16):
            #     _, _, pred_action, _ = lapo(label_obs_to_use, label_future_obs_to_use, predict_true_act=True)

            #     loss1 = F.mse_loss(pred_action, label_actions)
            
            # # Final loss calculation
            # loss = loss0 + config.labeled_loss_coef * loss1

            loss = loss0
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(disentangled_laom.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()
            if i % config.target_update_every == 0:
                soft_update(target_disentangled_laom, disentangled_laom, tau=config.target_tau)

            # update state probe (using disentangled_laom encoder)
            obs_v1_hidden = disentangled_laom.encode(obs_v1)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_states = state_probe(obs_v1_hidden.detach())
                state_probe_loss = F.mse_loss(pred_states, states)

            state_probe_optim.zero_grad(set_to_none=True)
            state_probe_loss.backward()
            state_probe_optim.step()

            # update action probe (using z_a from disentangled_laom)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action = act_linear_probe(z_a_v1.detach())
                act_probe_loss = F.mse_loss(pred_action, actions)

            act_probe_optim.zero_grad(set_to_none=True)
            act_probe_loss.backward()
            act_probe_optim.step()

            # update state-to-action probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                state_pred_action = state_act_linear_probe(obs_v1_hidden.detach())
                state_act_probe_loss = F.mse_loss(state_pred_action, actions)

            state_act_probe_optim.zero_grad(set_to_none=True)
            state_act_probe_loss.backward()
            state_act_probe_optim.step()

            # Prepare logging data
            log_data = {
                "disentangle/total_loss": loss.item(),
                "disentangle/con_act_loss": loss_con_act.item(),
                "disentangle/con_view_loss": loss_con_view.item(),
                "disentangle/zero_act_loss": loss_zero_act.item(),
                "disentangle/zero_view_loss": loss_zero_view.item(),
                "disentangle/recon_loss": loss_recon.item(),
                "disentangle/combined_loss": loss0.item(),
                "disentangle/state_probe_mse_loss": state_probe_loss.item(),
                "disentangle/action_probe_mse_loss": act_probe_loss.item(),
                "disentangle/state_action_probe_mse_loss": state_act_probe_loss.item(),
                "disentangle/throughput": total_tokens / (time.time() - start_time),
                "disentangle/learning_rate": scheduler.get_last_lr()[0],
                "disentangle/grad_norm": get_grad_norm(disentangled_laom).item(),
                "disentangle/z_a_norm": torch.norm(z_a_v1, p=2, dim=-1).mean().item(),
                "disentangle/z_v_norm": torch.norm(z_v_t1, p=2, dim=-1).mean().item(),
                "disentangle/z_a_std": z_a_v1.std(dim=0).mean().item(),
                "disentangle/z_v_std": z_v_t1.std(dim=0).mean().item(),
                "disentangle/epoch": epoch,
                "disentangle/total_steps": total_iterations,
                "disentangle/recon_loss_both": loss_recon_both.item(),
                }
            
            # Add coefficient logging
            log_data["disentangle/alpha_view_coef"] = alpha
            log_data["disentangle/beta_zero_act_coef"] = beta
            log_data["disentangle/gamma_zero_view_coef"] = gamma
            log_data["disentangle/delta_recon_coef"] = delta
            log_data["disentangle/contrastive_loss_coef"] = current_contrastive_coef
            
            wandb.log(log_data, step=step_counter)
            step_counter += 1
            
            # Reset fixed offset after processing the batch
            # if config.use_dtw_filter and config.fixed_batch_offset:
            #     dataloader.dataset.set_fixed_offset(None)

        # if config.eval_data_path is not None:
        #     eval_losses = evaluate(laom_action, act_linear_probe, eval_dataloader, config, device=DEVICE)
        #     log_data = {f"disentangle/{k}": v for k, v in eval_losses.items()}
        #     log_data["disentangle/epoch"] = epoch
        #     log_data["disentangle/total_steps"] = total_iterations
        #     wandb.log(log_data)
        
        
        # 10에폭마다 체크포인트 저장
        if (epoch + 1) % 400 == 0:
            # Save 3 models as dict
            models_dict = {
                'laom_action': laom_action.state_dict(),
                'laom_view': laom_view.state_dict(),
                'laom_shared': laom_shared.state_dict(),
            }
            torch.save({
                'models': models_dict,
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'loss': loss.item(),
                'config': config,
            }, os.path.join(checkpoint_dir, f"disentangled_epoch_{epoch+1}.pt"))

    # 최종 모델 저장
    torch.save({
        'disentangled_laom': disentangled_laom.state_dict(),
        'optimizer': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': config.num_epochs - 1,
        'loss': loss.item(),
        'config': config,
    }, os.path.join(checkpoint_dir, "disentangled_final.pt"))
    
    # Also save individual models for compatibility
    save_checkpoint(
        disentangled_laom,
        optim,
        scheduler,
        config.num_epochs - 1,
        loss.item(),
        os.path.join(checkpoint_dir, "lapo_final.pt"),
        config,
        config_path,
    )

    return lapo


def convert_tuples_to_lists(obj):
    """재귀적으로 dict/list 내의 모든 tuple을 list로 변환"""
    if isinstance(obj, dict):
        return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tuples_to_lists(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return obj


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, config=None, config_path=None):
    """모델 체크포인트를 저장합니다."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    # config 정보가 있으면 별도 yaml 파일로 저장
    if config is not None and config_path is not None:
        config_filepath = filepath.replace('.pt', '_config.yaml')
        # 원본 config 파일을 그대로 복사
        shutil.copy2(config_path, config_filepath)
        print(f"Config 저장됨: {config_filepath}")
    elif config is not None:
        # config_path가 없는 경우 기존 방식 사용
        config_filepath = filepath.replace('.pt', '_config.yaml')
        config_dict = convert_tuples_to_lists(asdict(config))
        with open(config_filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        print(f"Config 저장됨: {config_filepath}")
    
    torch.save(checkpoint_data, filepath)
    print(f"체크포인트 저장됨: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath):
    """모델 체크포인트를 불러옵니다."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {filepath}")
    
    try:
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        # Debugging: print shapes before loading
        if 'model_state_dict' in checkpoint and hasattr(model, 'true_actions_head'):
            print("--- Shape Debug ---")
            # Shape in the current model
            current_shape = model.true_actions_head.weight.shape
            print(f"Current model's 'true_actions_head' shape: {current_shape}")
            
            # Shape in the checkpoint
            checkpoint_shape = checkpoint['model_state_dict']['true_actions_head.weight'].shape
            print(f"Checkpoint's 'true_actions_head' shape: {checkpoint_shape}")
            print("-------------------")

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
        
    except Exception as e:
        raise RuntimeError(f"체크포인트 로딩 중 오류 발생: {filepath}, 에러: {e}")


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
                if isinstance(action_decoder, ActionDecoder):
                    action = action_decoder(obs_emb, action)
                else:
                    action = action_decoder(action)

            obs, reward, terminated, truncated, info = env.step(action.squeeze().cpu().numpy())
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


def train_bc(lam: DisentangledLAOM, config: BCConfig, checkpoint_dir: str, config_path: str = None):
    # Trajectory-level splits for BC stage
    bc_train_demos, bc_val_demos = get_or_create_traj_splits(
        hdf5_path=config.data_path,
        train_traj_num=config.train_traj_num,
        val_traj_num=config.val_traj_num,
        seed=config.seed,
        splits_dir="indices",
        dataset_type="labeled",
    )

    # Create train/val datasets restricted by selected demos
    # Note: BC stage does not use mixed_view_sampling
    train_dataset = DCSMVInMemoryDataset(
        config.data_path,
        max_offset=config.fixed_offset,
        frame_stack=config.frame_stack,
        device="cpu",
        camera_num=config.camera_num,
        view_keys=config.view_keys,
        resize_wh=(config.resize_size, config.resize_size),
        selected_demo_names=bc_train_demos,
        mixed_view_sampling=False,
        positive_samples_per_instance=4,
    )
    val_dataset = DCSMVInMemoryDataset(
        config.data_path,
        max_offset=config.fixed_offset,
        frame_stack=config.frame_stack,
        device="cpu",
        camera_num=config.camera_num,
        view_keys=config.view_keys,
        resize_wh=(config.resize_size, config.resize_size),
        selected_demo_names=bc_val_demos,
        mixed_view_sampling=False,
        positive_samples_per_instance=4,
    )

    collate_fn = partial(metric_learning_collate_fn, K=config.views_per_instance, mixed_view_sampling=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Set fixed offset for both datasets
    train_dataset.set_fixed_offset(config.fixed_offset)
    val_dataset.set_fixed_offset(config.fixed_offset)

    # Debug: Verify fixed offset is set correctly
    print(f"\n=== train_bc Dataset Debug ===")
    print(f"Requested fixed offset: {config.fixed_offset}")
    print(f"Dataset has set_fixed_offset method: {hasattr(train_dataset, 'set_fixed_offset')}")
    if hasattr(train_dataset, 'fixed_offset'):
        print(f"Train dataset fixed_offset: {train_dataset.fixed_offset}")
    if hasattr(val_dataset, 'fixed_offset'):
        print(f"Val dataset fixed_offset: {val_dataset.fixed_offset}")
    print("=" * 40)
    # eval_env = create_env_from_df(
    #     config.data_path,
    #     config.dcs_backgrounds_path,
    #     config.dcs_backgrounds_split,
    #     frame_stack=config.frame_stack,
    # )
    # print(eval_env.observation_space)
    # print(eval_env.action_space)

    num_actions = lam.latent_act_dim
    for p in lam.parameters():
        p.requires_grad_(False)
    lam.eval()

    actor = Actor(
        shape=(3 * config.frame_stack, train_dataset.img_hw, train_dataset.img_hw),
        num_actions=num_actions,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        dropout=config.dropout,
    ).to(DEVICE)

    optim = torch.optim.AdamW(params=get_optim_groups(actor, config.weight_decay), lr=config.learning_rate, fused=True)
    # scheduler setup
    total_updates = len(train_dataloader) * config.num_epochs
    warmup_updates = len(train_dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    # for debug
    print("Latent action dim:", num_actions)
    # Output dimension is fixed_offset * act_dim
    output_dim = config.fixed_offset * train_dataset.act_dim
    print("Output dim:", output_dim)
    
    # Debug: Verify output_dim calculation
    print(f"\n=== train_bc Output Dim Debug ===")
    print(f"Fixed offset: {config.fixed_offset}")
    print(f"Dataset act_dim: {train_dataset.act_dim}")
    print(f"Calculated output_dim: {output_dim}")
    print(f"Expected sequence length: {config.fixed_offset}")
    print(f"Expected flattened length: {(config.fixed_offset) * train_dataset.act_dim}")
    print(f"Match with expected: {output_dim == (config.fixed_offset) * train_dataset.act_dim}")
    print("=" * 50)
    
    act_decoder = nn.Sequential(
        nn.Linear(num_actions, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, output_dim)
    ).to(DEVICE)

    act_decoder_optim = torch.optim.AdamW(params=act_decoder.parameters(), lr=config.learning_rate, fused=True)
    act_decoder_scheduler = linear_annealing_with_warmup(act_decoder_optim, warmup_updates, total_updates)

    torchinfo.summary(actor, input_size=(1, 3 * config.frame_stack, train_dataset.img_hw, train_dataset.img_hw))
    if config.use_aug:
        augmenter = Augmenter(img_resolution=train_dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    step_counter = 0  # Add step counter for wandb logging
    for epoch in trange(config.num_epochs, desc="Epochs"):
        actor.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False):
            total_tokens += config.batch_size
            total_steps += 1

            # Multi-view data is now shaped (P*K, H, W, C) from collate_fn.
            # Permute to (P*K, C, H, W) for conv networks.
            obs, next_obs, future_obs, debug_actions, debug_action_sequences, debug_states, instance_ids, offsets, action_sequences_mask = [
                batch[k].to(DEVICE) for k in ["obs", "next_obs", "future_obs", "actions", "action_sequences", "states", "instance_ids", "offsets", "action_sequences_mask"]
            ]

            # Debug: Check offset consistency in train_bc
            if epoch == 0 and total_steps <= 3:  # First 3 batches only
                print(f"\n=== train_bc Debug: Batch {total_steps} ===")
                print(f"Fixed offset setting: {config.fixed_offset}")
                print(f"Actual offsets in batch: {offsets.cpu().numpy()}")
                print(f"All offsets same: {torch.all(offsets == offsets[0]).item()}")
                print(f"Expected sequence length: {config.fixed_offset}")
                print(f"Action sequences shape: {debug_action_sequences.shape}")
                print(f"Action sequences expected shape: ({obs.shape[0]}, {(config.fixed_offset) * train_dataset.act_dim})")
                print(f"Action sequences mask shape: {action_sequences_mask.shape}")
                print(f"Action sequences mask sum (actual lengths): {action_sequences_mask.sum(dim=1).cpu().numpy()}")
                print("=" * 50)
                
            # Save debug images from BC dataloader (first batch only)
            # Note: BC dataloader uses old structure (obs, next_obs, future_obs)
            # while unlabeled uses new view pair structure (obs_v1, obs_v2, etc.)
            # if epoch == 0 and total_steps == 1:
            #     print("\n=== Saving Debug Images from BC Training Dataloader ===")
            #     save_debug_images(
            #         obs, 
            #         next_obs, 
            #         future_obs,
            #         instance_ids,
            #         offsets,
            #         save_dir=os.path.join(checkpoint_dir, "debug_images_train_bc"),
            #         num_samples=min(24, obs.shape[0])
            #     )
            
            obs = obs.permute((0, 3, 1, 2))
            next_obs = next_obs.permute((0, 3, 1, 2))
            future_obs = future_obs.permute((0, 3, 1, 2))
            
            # Normalization is now applied to the already flattened batch
            obs = normalize_img(obs)
            next_obs = normalize_img(next_obs)
            future_obs = normalize_img(future_obs)

            # label with lapo latent actions
            target_actions = lam.predict_action(obs, future_obs)

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

            # optimizing the probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_true_actions = act_decoder(pred_actions.detach())
                # Use action_sequences for fixed offset
                decoder_loss = F.mse_loss(pred_true_actions, debug_action_sequences)

                # Debug: Check BC action decoder prediction vs target
                if epoch == 0 and total_steps <= 3:
                    print(f"\n--- train_bc Action Decoder Debug: Batch {total_steps} ---")
                    print(f"Predicted true actions shape: {pred_true_actions.shape}")
                    print(f"Target action sequences shape: {debug_action_sequences.shape}")
                    print(f"Shape match: {pred_true_actions.shape == debug_action_sequences.shape}")
                    
                    # Show first sample's prediction vs target
                    if pred_true_actions.shape[0] > 0:
                        pred_sample = pred_true_actions[0].detach().cpu().float().numpy()
                        target_sample = debug_action_sequences[0].detach().cpu().float().numpy()
                        print(f"BC first sample prediction (first 10 values): {pred_sample[:10]}")
                        print(f"BC first sample target (first 10 values): {target_sample[:10]}")
                        print(f"BC first sample MSE: {F.mse_loss(pred_true_actions[0:1], debug_action_sequences[0:1]).item():.6f}")
                        
                        # Reshape to show sequence structure
                        seq_len = config.fixed_offset
                        pred_reshaped = pred_sample.reshape(seq_len, -1)
                        target_reshaped = target_sample.reshape(seq_len, -1)
                        print(f"BC prediction reshaped ({seq_len} x {train_dataset.act_dim}):")
                        print(np.round(pred_reshaped, 3))
                        print(f"BC target reshaped ({seq_len} x {train_dataset.act_dim}):")
                        print(np.round(target_reshaped, 3))
                    print("-" * 60)

            act_decoder_optim.zero_grad(set_to_none=True)
            decoder_loss.backward()
            act_decoder_optim.step()
            act_decoder_scheduler.step()

            wandb.log(
                {
                    "bc/mse_loss": loss.item(),
                    "bc/throughput": total_tokens / (time.time() - start_time),
                    "bc/learning_rate": scheduler.get_last_lr()[0],
                    "bc/act_decoder_probe_mse_loss": decoder_loss.item(),
                    "bc/epoch": epoch,
                    "bc/total_steps": total_steps,
                },
                step=step_counter
            )
            step_counter += 1
        
        # Validation loss 계산
        actor.eval()
        val_loss = 0.0
        val_decoder_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_obs, val_next_obs, val_future_obs, val_debug_actions, val_debug_action_sequences, val_debug_states, val_instance_ids, val_offsets, val_action_sequences_mask = [
                    val_batch[k].to(DEVICE) for k in ["obs", "next_obs", "future_obs", "actions", "action_sequences", "states", "instance_ids", "offsets", "action_sequences_mask"]
                ]
                
                val_obs = val_obs.permute((0, 3, 1, 2))
                val_next_obs = val_next_obs.permute((0, 3, 1, 2))
                val_future_obs = val_future_obs.permute((0, 3, 1, 2))
                
                val_obs = normalize_img(val_obs)
                val_next_obs = normalize_img(val_next_obs)
                val_future_obs = normalize_img(val_future_obs)
                
                val_target_actions = lam.predict_action(val_obs, val_future_obs)
                
                with torch.autocast(DEVICE, dtype=torch.bfloat16):
                    val_pred_actions, _ = actor(val_obs)
                    val_batch_loss = F.mse_loss(val_pred_actions, val_target_actions)
                    
                    val_pred_true_actions = act_decoder(val_pred_actions)
                    val_batch_decoder_loss = F.mse_loss(val_pred_true_actions, val_debug_action_sequences)
                
                val_loss += val_batch_loss.item()
                val_decoder_loss += val_batch_decoder_loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        val_decoder_loss /= val_batches
        
        # Validation loss만 wandb에 로깅
        wandb.log({
            "bc/val_mse_loss": val_loss,
            "bc/val_act_decoder_probe_mse_loss": val_decoder_loss,
            "bc/epoch": epoch,
        }, step=step_counter)
        
        # 10에폭마다 체크포인트 저장
        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                actor,
                optim,
                scheduler,
                epoch,
                loss.item(),
                os.path.join(checkpoint_dir, f"bc_offset_{config.fixed_offset}_epoch_{epoch+1}.pt"),
                config,
                config_path,
            )

    # 최종 모델 저장
    save_checkpoint(
        actor,
        optim,
        scheduler,
        config.num_epochs - 1,
        loss.item(),
        os.path.join(checkpoint_dir, f"bc_offset_{config.fixed_offset}_final.pt"),
        config,
        config_path,
    )

    # actor.eval()
    # eval_returns = evaluate_bc(
    #     eval_env,
    #     actor,
    #     num_episodes=config.eval_episodes,
    #     seed=config.eval_seed,
    #     device=DEVICE,
    #     action_decoder=act_decoder,
    # )
    # wandb.log(
    #     {
    #         "bc/eval_returns_mean": eval_returns.mean(),
    #         "bc/eval_returns_std": eval_returns.std(),
    #         "bc/epoch": epoch,
    #         "bc/total_steps": total_steps,
    #     }
    # )

    return actor


def create_bc_transformer_decoder(config: DecoderConfig, dataset, actor: Actor):
    """
    Create BC Transformer decoder using robomimic
    """
    if config.bc_transformer_config_path is None:
        # Use default BC transformer config
        config_path = "/home/s2/youngjoonjeong/github/robomimic/robomimic/config/default_templates/bc_transformer.json"
    else:
        config_path = config.bc_transformer_config_path
    
    # Load robomimic config
    robomimic_config = config_factory(algo_name="bc", config_name="bc_transformer", config_file=config_path)
    
    # Modify config for our use case
    robomimic_config.train.data = config.data_path
    robomimic_config.train.num_epochs = config.total_updates // len(dataset) // config.batch_size
    robomimic_config.train.batch_size = config.batch_size
    robomimic_config.train.seed = config.eval_seed
    
    # Set observation shapes for our dataset
    obs_shapes = {
        "obs": {
            "low_dim": [dataset.state_dim],  # robot state dimension
            "rgb": [dataset.img_hw, dataset.img_hw, 3] if config.frame_stack == 1 else [config.frame_stack, dataset.img_hw, dataset.img_hw, 3]
        }
    }
    
    # Create BC Transformer
    bc_transformer = BC_Transformer(
        algo_config=robomimic_config.algo,
        obs_config=robomimic_config.observation,
        global_config=robomimic_config,
        obs_shapes=obs_shapes,
        ac_dim=dataset.act_dim,
        device=DEVICE
    )
    
    return bc_transformer


def train_act_decoder(actor: Actor, config: DecoderConfig, bc_config: BCConfig, checkpoint_dir: str, config_path: str = None):
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()

    # Trajectory-level splits for Decoder stage
    dec_train_demos, dec_val_demos = get_or_create_traj_splits(
        hdf5_path=config.data_path,
        train_traj_num=config.train_traj_num,
        val_traj_num=config.val_traj_num,
        seed=config.eval_seed,
        splits_dir="indices",
        dataset_type="labeled",
    )
    
    # Create train/val datasets restricted by selected demos
    # Note: Decoder stage does not use mixed_view_sampling
    train_dataset = DCSMVInMemoryDataset(
        config.data_path, 
        max_offset=config.fixed_offset, 
        frame_stack=config.frame_stack, 
        device="cpu",
        camera_num=config.camera_num,
        view_keys=config.view_keys,
        resize_wh=(config.resize_size, config.resize_size),
        selected_demo_names=dec_train_demos,
        mixed_view_sampling=False,
        positive_samples_per_instance=4,
    )
    val_dataset = DCSMVInMemoryDataset(
        config.data_path, 
        max_offset=config.fixed_offset, 
        frame_stack=config.frame_stack, 
        device="cpu",
        camera_num=config.camera_num,
        view_keys=config.view_keys,
        resize_wh=(config.resize_size, config.resize_size),
        selected_demo_names=dec_val_demos,
        mixed_view_sampling=False,
        positive_samples_per_instance=4,
    )
    collate_fn = partial(metric_learning_collate_fn, K=config.views_per_instance, mixed_view_sampling=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    # Set fixed offset
    train_dataset.set_fixed_offset(config.fixed_offset)
    val_dataset.set_fixed_offset(config.fixed_offset)
    
    # Debug: Verify fixed offset is set correctly
    print(f"\n=== train_act_decoder Dataset Debug ===")
    print(f"Requested fixed offset: {config.fixed_offset}")
    print(f"Dataset has set_fixed_offset method: {hasattr(train_dataset, 'set_fixed_offset')}")
    if hasattr(train_dataset, 'fixed_offset'):
        print(f"Train dataset fixed_offset: {train_dataset.fixed_offset}")
    if hasattr(val_dataset, 'fixed_offset'):
        print(f"Val dataset fixed_offset: {val_dataset.fixed_offset}")
    print("=" * 40)
    
    # Decoder는 traj 기반 단일 dataset으로 학습/검증을 동일 세트에서 진행하거나,
    # 필요 시 별도 val split을 원하면 상단에서 selected_demo_names를 val로 분기해 두 개 생성하면 됩니다.
    # 여기서는 단일 dataset과 dataloader만 사용합니다.
    print(f"Decoder train dataset size (moments): {len(train_dataset)}")
    print(f"Decoder val dataset size (moments): {len(val_dataset)}")
    num_epochs = max(1, config.total_updates // max(1, len(train_dataloader)))

    # Output dimension is fixed_offset * act_dim
    true_act_dim = config.fixed_offset * train_dataset.act_dim
    print("True act dim:", true_act_dim)
    
    # Debug: Verify true_act_dim calculation
    print(f"\n=== train_act_decoder True Action Dim Debug ===")
    print(f"Fixed offset: {config.fixed_offset}")
    print(f"Dataset act_dim: {train_dataset.act_dim}")
    print(f"Calculated true_act_dim: {true_act_dim}")
    print(f"Expected sequence length: {config.fixed_offset}")
    print(f"Expected flattened length: {(config.fixed_offset) * train_dataset.act_dim}")
    print(f"Match with expected: {true_act_dim == (config.fixed_offset) * train_dataset.act_dim}")
    print("=" * 50)
    
    # Choose action decoder type
    if config.action_decoder_type == "bc_transformer":
        if not ROBOMIMIC_AVAILABLE:
            raise ImportError("Robomimic is not available. Please install robomimic or use action_decoder_type='simple'")
        
        print("Using BC Transformer as action decoder")
        action_decoder = create_bc_transformer_decoder(config, train_dataset, actor)
    else:
        print("Using simple ActionDecoder")
        action_decoder = ActionDecoder(
            obs_emb_dim=math.prod(actor.final_encoder_shape),
            latent_act_dim=actor.num_actions,
            true_act_dim=true_act_dim,
            hidden_dim=config.hidden_dim,
        ).to(DEVICE)

    # Add Direct Image Decoder for comparison
    print("Adding Direct Image Decoder for comparison")
    direct_decoder = DirectImageDecoder(
        input_shape=(3 * config.frame_stack, train_dataset.img_hw, train_dataset.img_hw),
        true_act_dim=true_act_dim,
        hidden_dim=config.hidden_dim,
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        params=get_optim_groups(action_decoder, config.weight_decay), lr=config.learning_rate, fused=True
    )
    
    # Optimizer for direct decoder
    direct_optim = torch.optim.AdamW(
        params=get_optim_groups(direct_decoder, config.weight_decay), lr=config.learning_rate, fused=True
    )
    # eval_env = create_env_from_df(
    #     config.data_path,
    #     config.dcs_backgrounds_path,
    #     config.dcs_backgrounds_split,
    #     frame_stack=config.frame_stack,
    # )
    # print(eval_env.observation_space)
    # print(eval_env.action_space)

    # scheduler setup
    total_updates = len(train_dataloader) * num_epochs
    warmup_updates = len(train_dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)
    direct_scheduler = linear_annealing_with_warmup(direct_optim, warmup_updates, total_updates)

    if config.use_aug:
        augmenter = Augmenter(img_resolution=train_dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    step_counter = 0  # Add step counter for wandb logging

    for epoch in trange(num_epochs, desc="Epochs"):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            total_tokens += config.batch_size
            total_steps += 1

            # Multi-view data is now shaped (P*K, H, W, C) from collate_fn.
            # Permute to (P*K, C, H, W) for conv networks.
            obs, next_obs, future_obs, debug_actions, debug_action_sequences, debug_states, instance_ids, offsets, action_sequences_mask = [
                batch[k].to(DEVICE) for k in ["obs", "next_obs", "future_obs", "actions", "action_sequences", "states", "instance_ids", "offsets", "action_sequences_mask"]
            ]

            # Debug: Check offset consistency in train_act_decoder
            if epoch == 0 and total_steps <= 3:  # First 3 batches only
                print(f"\n=== train_act_decoder Debug: Batch {total_steps} ===")
                print(f"Fixed offset setting: {config.fixed_offset}")
                print(f"Actual offsets in batch: {offsets.cpu().numpy()}")
                print(f"All offsets same: {torch.all(offsets == offsets[0]).item()}")
                print(f"Expected sequence length: {config.fixed_offset}")
                print(f"Action sequences shape: {debug_action_sequences.shape}")
                print(f"Action sequences expected shape: ({obs.shape[0]}, {(config.fixed_offset) * train_dataset.act_dim})")
                print(f"Action sequences mask shape: {action_sequences_mask.shape}")
                print(f"Action sequences mask sum (actual lengths): {action_sequences_mask.sum(dim=1).cpu().numpy()}")
                print(f"True act dim (output): {true_act_dim}")
                
                # # Fixed offset sequence 검증
                # if torch.all(offsets == offsets[0]).item():
                #     expected_seq_len = offsets[0].item() + 1  # offset + 1 = sequence length
                #     print(f"✓ All samples have same offset: {offsets[0].item()}")
                #     print(f"✓ Expected sequence length: {expected_seq_len}")
                #     print(f"✓ Expected flattened length: {expected_seq_len * dataset.act_dim}")
                #     print(f"✓ Actual flattened length: {debug_action_sequences.shape[1]}")
                #     print(f"✓ Shape match: {debug_action_sequences.shape[1] == expected_seq_len * dataset.act_dim}")
                # else:
                #     print(f"⚠ Warning: Offsets vary in batch!")
                #     print(f"  Offset range: {offsets.min().item()} - {offsets.max().item()}")
                #     print(f"  Unique offsets: {torch.unique(offsets).cpu().numpy()}")
                
                print("=" * 50)
            
            obs = obs.permute((0, 3, 1, 2))
            # rescale from 0..255 -> -1..1
            obs = normalize_img(obs)

            if config.use_aug:
                obs = augmenter(obs)

            # Train Latent Action Decoder
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                with torch.no_grad():
                    latent_actions, obs_emb = actor(obs)
                pred_actions_latent = action_decoder(obs_emb, latent_actions)
                loss_latent = F.mse_loss(pred_actions_latent, debug_action_sequences)

                # Debug: Check prediction vs target shapes and values
                if epoch == 0 and total_steps <= 3:
                    print(f"\n--- train_act_decoder Prediction Debug: Batch {total_steps} ---")
                    print(f"Predicted actions shape: {pred_actions_latent.shape}")
                    print(f"Target actions shape: {debug_action_sequences.shape}")
                    print(f"Shape match: {pred_actions_latent.shape == debug_action_sequences.shape}")
                    
                    # Show first sample's prediction vs target
                    if pred_actions_latent.shape[0] > 0:
                        pred_sample = pred_actions_latent[0].detach().cpu().float().numpy()
                        target_sample = debug_action_sequences[0].detach().cpu().float().numpy()
                        print(f"First sample prediction (first 10 values): {pred_sample[:10]}")
                        print(f"First sample target (first 10 values): {target_sample[:10]}")
                        print(f"First sample MSE: {F.mse_loss(pred_actions_latent[0:1], debug_action_sequences[0:1]).item():.6f}")
                        
                        # Reshape to show sequence structure
                        seq_len = config.fixed_offset
                        pred_reshaped = pred_sample.reshape(seq_len, -1)
                        target_reshaped = target_sample.reshape(seq_len, -1)
                        print(f"Prediction reshaped ({seq_len} x {train_dataset.act_dim}):")
                        print(np.round(pred_reshaped, 3))
                        print(f"Target reshaped ({seq_len} x {train_dataset.act_dim}):")
                        print(np.round(target_reshaped, 3))
                    print("-" * 60)

            optim.zero_grad(set_to_none=True)
            loss_latent.backward()
            optim.step()
            scheduler.step()

            # Train Direct Image Decoder
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_actions_direct = direct_decoder(obs)
                loss_direct = F.mse_loss(pred_actions_direct, debug_action_sequences)

                # Debug: Check direct decoder prediction vs target
                if epoch == 0 and total_steps <= 3:
                    print(f"\n--- train_act_decoder Direct Decoder Debug: Batch {total_steps} ---")
                    print(f"Direct prediction shape: {pred_actions_direct.shape}")
                    print(f"Target actions shape: {debug_action_sequences.shape}")
                    print(f"Shape match: {pred_actions_direct.shape == debug_action_sequences.shape}")
                    
                    # Show first sample's prediction vs target
                    if pred_actions_direct.shape[0] > 0:
                        pred_sample = pred_actions_direct[0].detach().cpu().float().numpy()
                        target_sample = debug_action_sequences[0].detach().cpu().float().numpy()
                        print(f"Direct first sample prediction (first 10 values): {pred_sample[:10]}")
                        print(f"Direct first sample target (first 10 values): {target_sample[:10]}")
                        print(f"Direct first sample MSE: {F.mse_loss(pred_actions_direct[0:1], debug_action_sequences[0:1]).item():.6f}")
                        
                        # Reshape to show sequence structure
                        seq_len = config.fixed_offset
                        pred_reshaped = pred_sample.reshape(seq_len, -1)
                        target_reshaped = target_sample.reshape(seq_len, -1)
                        print(f"Direct prediction reshaped ({seq_len} x {train_dataset.act_dim}):")
                        print(np.round(pred_reshaped, 3))
                        print(f"Direct target reshaped ({seq_len} x {train_dataset.act_dim}):")
                        print(np.round(target_reshaped, 3))
                    print("-" * 60)

            direct_optim.zero_grad(set_to_none=True)
            loss_direct.backward()
            direct_optim.step()
            direct_scheduler.step()

            wandb.log(
                {
                    "decoder/latent_mse_loss": loss_latent.item(),
                    "decoder/direct_mse_loss": loss_direct.item(),
                    "decoder/latent_vs_direct_ratio": loss_latent.item() / (loss_direct.item() + 1e-8),
                    "decoder/throughput": total_tokens / (time.time() - start_time),
                    "decoder/learning_rate": scheduler.get_last_lr()[0],
                    "decoder/epoch": epoch,
                    "decoder/total_steps": total_steps,
                },
                step=step_counter
            )
            step_counter += 1
        
        # Validation loss 계산
        action_decoder.eval()
        direct_decoder.eval()
        val_latent_loss = 0.0
        val_direct_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_obs, val_next_obs, val_future_obs, val_debug_actions, val_debug_action_sequences, val_debug_states, val_instance_ids, val_offsets, val_action_sequences_mask = [
                    val_batch[k].to(DEVICE) for k in ["obs", "next_obs", "future_obs", "actions", "action_sequences", "states", "instance_ids", "offsets", "action_sequences_mask"]
                ]
                
                val_obs = val_obs.permute((0, 3, 1, 2))
                val_obs = normalize_img(val_obs)
                
                # Latent Action Decoder validation
                with torch.autocast(DEVICE, dtype=torch.bfloat16):
                    val_latent_actions, val_obs_emb = actor(val_obs)
                    val_pred_actions_latent = action_decoder(val_obs_emb, val_latent_actions)
                    val_batch_latent_loss = F.mse_loss(val_pred_actions_latent, val_debug_action_sequences)
                
                # Direct Image Decoder validation
                with torch.autocast(DEVICE, dtype=torch.bfloat16):
                    val_pred_actions_direct = direct_decoder(val_obs)
                    val_batch_direct_loss = F.mse_loss(val_pred_actions_direct, val_debug_action_sequences)
                
                val_latent_loss += val_batch_latent_loss.item()
                val_direct_loss += val_batch_direct_loss.item()
                val_batches += 1
        
        val_latent_loss /= val_batches
        val_direct_loss /= val_batches
        
        # Validation loss를 wandb에 로깅
        wandb.log({
            "decoder/val_latent_mse_loss": val_latent_loss,
            "decoder/val_direct_mse_loss": val_direct_loss,
            "decoder/val_latent_vs_direct_ratio": val_latent_loss / (val_direct_loss + 1e-8),
            "decoder/epoch": epoch,
        }, step=step_counter)
        
        # Training mode로 복원
        action_decoder.train()
        direct_decoder.train()
        
        # 10에폭마다 체크포인트 저장
        if (epoch + 1) % 2500 == 0:
            # Latent Action Decoder 체크포인트
            save_checkpoint(
                action_decoder,
                optim,
                scheduler,
                epoch,
                loss_latent.item(),
                os.path.join(checkpoint_dir, f"action_decoder_epoch_{epoch+1}.pt"),
                config,
                config_path,
            )
            # Direct Image Decoder 체크포인트
            save_checkpoint(
                direct_decoder,
                direct_optim,
                direct_scheduler,
                epoch,
                loss_direct.item(),
                os.path.join(checkpoint_dir, f"direct_decoder_epoch_{epoch+1}.pt"),
                config,
                config_path,
            )

    # 최종 모델 저장
    save_checkpoint(
        action_decoder,
        optim,
        scheduler,
        num_epochs - 1,
        loss_latent.item(),
        os.path.join(checkpoint_dir, "action_decoder_final.pt"),
        config,
        config_path,
    )
    save_checkpoint(
        direct_decoder,
        direct_optim,
        direct_scheduler,
        num_epochs - 1,
        loss_direct.item(),
        os.path.join(checkpoint_dir, "direct_decoder_final.pt"),
        config,
        config_path,
    )

    actor.eval()
    # eval_returns = evaluate_bc(
    #     eval_env,
    #     actor,
    #     num_episodes=config.eval_episodes,
    #     seed=config.eval_seed,
    #     device=DEVICE,
    #     action_decoder=action_decoder,
    # )
    # wandb.log(
    #     {
    #         "decoder/eval_returns_mean": eval_returns.mean(),
    #         "decoder/eval_returns_std": eval_returns.std(),
    #         "decoder/epoch": epoch,
    #         "decoder/total_steps": total_steps,
    #     }
    # )

    return action_decoder, direct_decoder


@pyrallis.wrap()
def train(config: Config, config_path: str = None):
    set_seed(config.seed)

    # 체크포인트 디렉토리 및 wandb 실행 이름 설정
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.lapo_checkpoint_path:
        checkpoint_dir = os.path.dirname(config.lapo_checkpoint_path)
        run_name = config.run_name if config.run_name else os.path.basename(checkpoint_dir)
    elif config.bc_checkpoint_path:
        checkpoint_dir = os.path.dirname(config.bc_checkpoint_path)
        run_name = config.run_name if config.run_name else os.path.basename(checkpoint_dir)
    else:
        if config.run_name:
            checkpoint_dir = os.path.join("/shared/s2/lab01/youngjoonjeong/multiview/latent-action-view-action", config.run_name)
            run_name = config.run_name
        else:
            checkpoint_dir = os.path.join("/shared/s2/lab01/youngjoonjeong/multiview/latent-action-view-action", timestamp)
            run_name = timestamp
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Run name: {run_name}")


    print("--- Running Dataloader Shape Check ---")
    lapo_config = config.lapo
    
    # --- Unlabeled Dataset Check ---
    print("\n--- Unlabeled Dataset ---")
    unlabeled_batch = None
    try:
        unlabeled_dataset = DCSViewActionInMemoryDataset(
            lapo_config.data_path,
            max_offset=lapo_config.future_obs_offset,
            frame_stack=lapo_config.frame_stack,
            device="cpu",
            camera_num=lapo_config.camera_num,
            view_keys=lapo_config.view_keys,
            resize_wh=(lapo_config.resize_size, lapo_config.resize_size),
        )
        collate_fn = partial(view_action_collate_fn, K=lapo_config.views_per_instance)
        unlabeled_dataloader = DataLoader(
            unlabeled_dataset, batch_size=lapo_config.batch_size, shuffle=True, collate_fn=collate_fn
        )
        print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
        print(f"Unlabeled dataloader P (batch_size): {lapo_config.batch_size}, K (views_per_instance): {lapo_config.views_per_instance}")
        print(f"Available views per instance: {unlabeled_dataset.view_keys_to_load}")
        print(f"Number of available views: {len(unlabeled_dataset.view_keys_to_load)}")
        
        # --- Set fixed offset for debugging if enabled ---
        if config.debug_dataloader and lapo_config.fixed_batch_offset:
            # debug_offset = torch.randint(5, lapo_config.future_obs_offset + 1, (1,)).item()
            debug_offset = 10
            unlabeled_dataloader.dataset.set_fixed_offset(debug_offset)
            print(f"\n=== Debug DTW: Using fixed offset for batch: {debug_offset} ===")

        unlabeled_batch = next(iter(unlabeled_dataloader))
        if unlabeled_batch:
            print("INFO: unlabeled batch keys: ", unlabeled_batch.keys())
            obs = unlabeled_batch["obs_v1"]
            print(f"  Batch obs shape (B = P * K): {obs.shape}, dtype: {obs.dtype}")
            print("-" * 25)

    except Exception as e:
        print(f"An error occurred during unlabeled dataloader check: {e}")

    # # --- Labeled Dataset Check ---
    # print("\n--- Labeled Dataset ---")
    # labeled_batch = None
    # try:
    #     labeled_dataset = DCSMVTrueActionsDataset(
    #         lapo_config.labeled_data_path,
    #         max_offset=lapo_config.future_obs_offset,
    #         frame_stack=lapo_config.frame_stack,
    #         device="cpu",
    #         camera_num=lapo_config.camera_num,
    #         view_keys=lapo_config.view_keys,
    #         resize_wh=(lapo_config.resize_size, lapo_config.resize_size)
    #     )
    #     collate_fn = partial(metric_learning_collate_fn, K=lapo_config.views_per_instance)
    #     labeled_dataloader = DataLoader(
    #         labeled_dataset, batch_size=lapo_config.labeled_batch_size, collate_fn=collate_fn
    #     )
    #     print(f"Labeled dataset size: (IterableDataset)")
    #     print(f"Labeled dataloader P (batch_size): {lapo_config.labeled_batch_size}, K (views_per_instance): {lapo_config.views_per_instance}")
        
    #     labeled_batch = next(iter(labeled_dataloader))
    #     if labeled_batch:
    #         obs = labeled_batch["obs"]
    #         print(f"  Batch obs shape (B = P * K): {obs.shape}, dtype: {obs.dtype}")
    #         print("-" * 25)
            
    # except Exception as e:
    #     print(f"An error occurred during labeled dataloader check: {e}")

    if config.debug_dataloader:
        print("\n--- Debug Mode: Printing batch details ---")
        
        # Unlabeled batch 상세 출력
        print("\n=== Unlabeled Batch Details ===")
        if unlabeled_batch:
            for key, value in unlabeled_batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                    if key in ["obs_v1", "obs_v2", "future_obs_v1", "future_obs_v2"]:
                        # uint8 타입의 경우 float로 변환 후 통계 계산
                        if value.dtype == torch.uint8:
                            value_float = value.float()
                            print(f"  - min: {value_float.min().item():.4f}, max: {value_float.max().item():.4f}, mean: {value_float.mean().item():.4f}")
                        else:
                            print(f"  - min: {value.min().item():.4f}, max: {value.max().item():.4f}, mean: {value.mean().item():.4f}")
                    elif key in ["instance_ids", "offsets", "view_pair_ids"]:
                        # instance_ids, offsets, view_pair_ids는 실제 값 출력
                        print(f"  - values: {value}")
                        if value.numel() <= 20:  # 값이 적으면 모두 출력
                            print(f"  - all values: {value.tolist()}")
                        else:  # 값이 많으면 처음 10개와 마지막 10개만 출력
                            print(f"  - first 10: {value[:10].tolist()}")
                            print(f"  - last 10: {value[-10:].tolist()}")
                else:
                    print(f"{key}: {type(value)} = {value}")
            num_debug_samples = 24
            # 이미지 저장
            print("\n=== Saving Debug Images ===")
            # Recover view_ids from view_pair_ids
            view_ids_v1 = unlabeled_batch["view_pair_ids"] // 10000
            view_ids_v2 = unlabeled_batch["view_pair_ids"] % 10000
            save_debug_images(
                unlabeled_batch["obs_v1"], 
                unlabeled_batch["obs_v2"], 
                unlabeled_batch["future_obs_v1"],
                unlabeled_batch["future_obs_v2"],
                view_ids_v1,
                view_ids_v2,
                unlabeled_batch["instance_ids"],
                unlabeled_batch["offsets"],
                save_dir=os.path.join(checkpoint_dir, "debug_images_unlabeled"),
                num_samples=num_debug_samples
            )
            
            # --- DTW Debugging ---
            print(f"\n=== DTW Debugging for First {num_debug_samples} Samples ===")
            
            if unlabeled_batch["action_sequences"].shape[0] >= num_debug_samples:
                action_sequences = unlabeled_batch["action_sequences"][:num_debug_samples]
                offsets = unlabeled_batch["offsets"][:num_debug_samples]
                action_dim = unlabeled_dataset.act_dim

                # Check if all offsets in the batch are the same
                all_offsets_are_same = torch.all(offsets == offsets[0]).item()
                dtw_distances = None

                if all_offsets_are_same:
                    seq_len = offsets[0].item() + 1
                    print(f"--- All samples have fixed sequence length: {seq_len} ---")
                    action_sequences_np = action_sequences.cpu().numpy().astype(np.float64)
                    
                    expected_len = seq_len * action_dim
                    actual_len = action_sequences_np.shape[1]

                    if actual_len == expected_len:
                        action_sequences_reshaped_np = action_sequences_np.reshape(-1, seq_len, action_dim)
                        
                        print("\n--- Original Action Sequences (for DTW) ---")
                        print(np.round(action_sequences_reshaped_np, 2))
                        
                        # --- Per-Dimension Analysis & Normalization ---
                        num_samples, seq_len, action_dim = action_sequences_reshaped_np.shape

                        # 1. Check stddev of each action dimension BEFORE normalization to verify hypothesis
                        std_per_dim = np.std(action_sequences_reshaped_np.reshape(-1, action_dim), axis=0)
                        print("\n--- Stddev per Action Dimension (Original) ---")
                        print(np.round(std_per_dim, 4))

                        # 2. Normalize each dimension independently using Min-Max scaling to [-1, 1]
                        min_per_dim = np.min(action_sequences_reshaped_np, axis=(0, 1), keepdims=True)
                        max_per_dim = np.max(action_sequences_reshaped_np, axis=(0, 1), keepdims=True)
                        range_per_dim = max_per_dim - min_per_dim
                        # Avoid division by zero for dimensions with no variance
                        range_per_dim[range_per_dim == 0] = 1 
                        
                        normalized_sequences = 2 * ((action_sequences_reshaped_np - min_per_dim) / range_per_dim) - 1
                        
                        print("\n--- Action Sequences (Per-Dimension Min-Max Scaled to [-1, 1]) ---")
                        print(np.round(normalized_sequences, 2))

                        # --- Calculate All Distance Matrices using NORMALIZED data ---
                        # 1. DTW Distance (Time-series aware)
                        dtw_distances = cdist_dtw(normalized_sequences)
                        
                        # For Cosine and L2, flatten the normalized sequences into single vectors
                        flattened_sequences = normalized_sequences.reshape(num_samples, seq_len * action_dim)

                        # 2. L2 (Euclidean) Distance (on flattened vectors)
                        l2_distances = cdist(flattened_sequences, flattened_sequences, metric='euclidean')

                        # 3. Cosine Distance (on flattened vectors)
                        cosine_distances = cdist(flattened_sequences, flattened_sequences, metric='cosine')

                        print("\n--- DTW Distance Matrix (tslearn) ---")
                        print(np.round(dtw_distances, 2))

                        print("\n--- L2 Distance Matrix (scipy, flattened) ---")
                        print(np.round(l2_distances, 2))

                        print("\n--- Cosine Distance Matrix (scipy, flattened) ---")
                        print(np.round(cosine_distances, 2))

                    else:
                        print(f"Action sequence shape mismatch. Expected flat length {expected_len}, got {actual_len}. Cannot calculate DTW matrix.")
                        
                else:
                    # Fallback to pairwise calculation if offsets are not the same
                    print("--- Offsets vary in batch, calculating DTW pairwise ---")
                    print("--- Action Sequences (Reshaped) ---")
                    action_sequences_reshaped_list = []

                    for i in range(num_debug_samples):
                        seq_len = offsets[i].item() + 1
                        expected_len = seq_len * action_dim
                        actual_len = action_sequences[i].shape[0]

                        if actual_len == expected_len:
                            action_seq_reshaped = action_sequences[i].reshape(seq_len, action_dim)
                            action_sequences_reshaped_list.append(action_seq_reshaped.cpu().numpy())
                            print(f"Sample {i:02d} (instance_id: {unlabeled_batch['instance_ids'][i].item()}, offset: {offsets[i].item()}):")
                            print(np.round(action_seq_reshaped.cpu().numpy(), 2))
                        else:
                            print(f"Sample {i:02d} (instance_id: {unlabeled_batch['instance_ids'][i].item()}) has shape mismatch. Expected {expected_len}, got {actual_len}. Skipping.")

                    if len(action_sequences_reshaped_list) > 1:
                        num_valid = len(action_sequences_reshaped_list)
                        dtw_distances_pairwise = np.full((num_valid, num_valid), np.inf)
                        
                        for i in range(num_valid):
                            for j in range(i, num_valid):
                                dist = cdist_dtw(action_sequences_reshaped_list[i].astype(np.float64), action_sequences_reshaped_list[j].astype(np.float64), metric="euclidean")
                                dtw_distances_pairwise[i, j] = dist
                                dtw_distances_pairwise[j, i] = dist
                        
                        dtw_distances = dtw_distances_pairwise
                        print("\n--- DTW Distance Matrix (pairwise) ---")
                        print(np.round(dtw_distances, 2))

                if dtw_distances is not None:
                    # Set diagonal to a large value to ignore it for min/max calculation
                    np.fill_diagonal(dtw_distances, np.inf)
                    
                    dtw_min = np.min(dtw_distances)
                    # Check if there are any finite values left before calculating max
                    if np.any(np.isfinite(dtw_distances)):
                        dtw_max = np.max(dtw_distances[np.isfinite(dtw_distances)])
                    else:
                        dtw_max = 0.0 # Or some other default value

                    print("\n--- DTW Statistics (excluding self-similarity) ---")
                    print(f"Min DTW Distance: {dtw_min:.4f}")
                    print(f"Max DTW Distance: {dtw_max:.4f}")

        print("\n--- Debug Mode: Exiting after batch inspection. ---")
        return # Exit after debugging

    # # Pre-generate traj splits for requested sizes before any training
    # preset_train_sizes = [2, 4, 8, 16, 32, 64, 128]
    # preset_val_size = 4
    # try:
    #     targets = [
    #         (config.lapo.data_path, "lapo", "unlabeled"),
    #         (config.lapo.labeled_data_path, "lapo", "labeled"),
    #         (config.bc.data_path, "bc", "labeled"),
    #         (config.decoder.data_path, "decoder", "labeled"),
    #     ]
    #     for hdf, stage, dtype in targets:
    #         for trn in preset_train_sizes:
    #             get_or_create_traj_splits(
    #                 hdf5_path=hdf,
    #                 train_traj_num=trn,
    #                 val_traj_num=preset_val_size,
    #                 seed=config.seed,
    #                 splits_dir="indices",
    #                 dataset_type=dtype,
    #             )
    # except Exception as e:
    #     print(f"Pre-generation of traj splits failed: {e}")

    run = wandb.init(
        project=config.project,
        group=config.group,
        name=run_name,
        config=asdict(config),
        save_code=True,
    )

    

    # stage 1: pretraining lapo on unlabeled dataset
    if config.lapo_checkpoint_path:
        print("=== Stage 1: LAPO Pretraining (skipped, loading from checkpoint) ===")
        # To create model, we need dataset metadata
        dataset = DCSMVInMemoryDataset(
            config.lapo.data_path, 
            max_offset=config.lapo.future_obs_offset, 
            frame_stack=config.lapo.frame_stack, 
            device="cpu",
            camera_num=config.lapo.camera_num,
            view_keys=config.lapo.view_keys,
            resize_wh=(config.lapo.resize_size, config.lapo.resize_size),
            mixed_view_sampling=config.lapo.mixed_view_sampling,
            positive_samples_per_instance=config.lapo.positive_samples_per_instance,
        )
        lapo = DisentangledLAOM(
            shape=(3 * config.lapo.frame_stack, dataset.img_hw, dataset.img_hw),
            latent_act_dim=config.lapo.latent_action_dim,
            latent_view_dim=config.lapo.latent_view_dim,
            encoder_scale=config.lapo.encoder_scale,
            encoder_channels=(16, 32, 64, 128, 256) if config.lapo.encoder_deep else (16, 32, 32),
            encoder_num_res_blocks=config.lapo.encoder_num_res_blocks,
            encoder_dropout=config.lapo.encoder_dropout,
            encoder_norm_out=config.lapo.encoder_norm_out,
            act_head_dim=config.lapo.act_head_dim,
            act_head_dropout=config.lapo.act_head_dropout,
            obs_head_dim=config.lapo.obs_head_dim,
            obs_head_dropout=config.lapo.obs_head_dropout,
            separate_fdm_heads=config.lapo.separate_fdm_heads,
        ).to(DEVICE)
        load_checkpoint(lapo, None, None, config.lapo_checkpoint_path)

    else:
        # run 시작 시점에 config 즉시 저장
        try:
            run_config_path = os.path.join(checkpoint_dir, "config.yaml")
            if config_path is not None and os.path.exists(config_path):
                shutil.copy2(config_path, run_config_path)
            else:
                config_dict = convert_tuples_to_lists(asdict(config))
                with open(run_config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            print(f"Run config 저장됨: {run_config_path}")
        except Exception as e:
            print(f"Run config 저장 실패: {e}")
        print("=== Stage 1: LAPO Pretraining ===")
        lapo = train_mv(config=config.lapo, checkpoint_dir=checkpoint_dir, config_path=config_path)
    
    # stage 2: pretraining bc on latent actions
    if config.bc_checkpoint_path:
        print("=== Stage 2: BC Pretraining (skipped, loading from checkpoint) ===")
        # We need dataset metadata to create the actor model
        # Note: BC stage does not use mixed_view_sampling
        dataset = DCSMVInMemoryDataset(
            config.bc.data_path, 
            max_offset=config.bc.fixed_offset, 
            frame_stack=config.bc.frame_stack, 
            device="cpu",
            camera_num=config.bc.camera_num,
            view_keys=config.bc.view_keys,
            resize_wh=(config.bc.resize_size, config.bc.resize_size),
            mixed_view_sampling=False,
            positive_samples_per_instance=4,
        )
        actor = Actor(
            shape=(3 * config.bc.frame_stack, dataset.img_hw, dataset.img_hw),
            num_actions=config.lapo.latent_action_dim,
            encoder_scale=config.bc.encoder_scale,
            encoder_channels=(16, 32, 64, 128, 256) if config.bc.encoder_deep else (16, 32, 32),
            encoder_num_res_blocks=config.bc.encoder_num_res_blocks,
            dropout=config.bc.dropout,
        ).to(DEVICE)
        load_checkpoint(actor, None, None, config.bc_checkpoint_path)
    else:
        print("=== Stage 2: BC Pretraining ===")
        actor = train_bc(lam=lapo, config=config.bc, checkpoint_dir=checkpoint_dir, config_path=config_path)
    
    # stage 3: finetune on labeled ground-truth actions
    print("=== Stage 3: Action Decoder Finetuning ===")
    action_decoder, direct_decoder = train_act_decoder(actor=actor, config=config.decoder, bc_config=config.bc, checkpoint_dir=checkpoint_dir, config_path=config_path)

    run.finish()
    return lapo

if __name__ == "__main__":
    train()

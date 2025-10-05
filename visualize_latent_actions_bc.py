import argparse
import math
import os
import yaml
from typing import List, Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
import umap
from sklearn.cluster import KMeans

from src.nn import LAOMWithLabels, Actor
from src.utils import DCSMVInMemoryDataset, normalize_img

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(model, filepath):
    """모델 체크포인트를 불러옵니다."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"체크포인트 불러옴: {filepath} (epoch {checkpoint.get('epoch', 'N/A')})")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")


@torch.no_grad()
def visualize(args):
    """
    지정된 궤적의 뷰별 잠재 행동을 UMAP으로 시각화합니다.
    - 색상: 뷰 (View)
    - 모양: 실제 행동 그룹 (Action Bin)
    """
    print("--- 1. 설정 파일 로딩 ---")
    # 체크포인트 경로 기준으로 같은 폴더의 config.yaml만 사용
    ckpt_path = args.checkpoint_path
    ckpt_dir = os.path.dirname(ckpt_path)
    config_path_to_use = os.path.join(ckpt_dir, 'config.yaml')

    if not os.path.exists(config_path_to_use):
        raise FileNotFoundError(f"체크포인트 폴더에서 config.yaml을 찾을 수 없습니다: {config_path_to_use}")

    print(f"Config 사용 경로: {config_path_to_use}")
    with open(config_path_to_use, 'r') as f:
        config_from_yaml = yaml.safe_load(f)
    
    lapo_config = config_from_yaml.get('lapo', {})
    if not lapo_config:
        raise ValueError(f"'{config_path_to_use}' 파일에 'lapo' 섹션이 없습니다.")
    # BC 설정이 존재하면 Actor 초기화에 사용, 없으면 lapo 설정으로 폴백
    bc_config = config_from_yaml.get('bc', {})
    
    # data_path는 인자로 받은 값으로 덮어씀
    lapo_config['data_path'] = args.data_path

    print("\n--- 2. 데이터셋 로딩 ---")
    if 'view_keys' not in lapo_config or not lapo_config['view_keys']:
        lapo_config['view_keys'] = [f"view_{i:02d}/agentview_image" for i in range(lapo_config.get('camera_num', 8))]
    elif isinstance(lapo_config['view_keys'][0], int):
        lapo_config['view_keys'] = [f"view_{i:02d}/agentview_image" for i in lapo_config['view_keys']]

    # 뷰 키 설정 - argparse에서 받은 값 사용
    if args.views == "all":
        # 0부터 20까지 모든 뷰 사용
        lapo_config['view_keys'] = [f"view_{i:02d}/agentview_image" for i in range(21)]
    else:
        # 지정된 뷰들만 사용
        view_indices = [int(v) for v in args.views.split(',')]
        lapo_config['view_keys'] = [f"view_{i:02d}/agentview_image" for i in view_indices]
    dataset = DCSMVInMemoryDataset(
        hdf5_path=lapo_config['data_path'],
        max_offset=lapo_config['future_obs_offset'],
        frame_stack=lapo_config['frame_stack'],
        device="cpu",
        view_keys=lapo_config['view_keys'],
         resize_wh=(64, 64),
        num_trajectories_to_load=200  # 모든 200개 궤적 로드
    )
    
    # 이제 데이터셋에 로드된 궤적 수가 실제 처리할 궤적 수가 됩니다.
    num_traj_to_process = len(dataset.traj_lens)
    print(f"사용 가능 뷰: {dataset.view_keys}")
    print(f"총 궤적 수 (로드됨): {num_traj_to_process}")


    print("\n--- 3. 모델 초기화 및 체크포인트 로딩 ---")
    # 공통 입력 shape
    input_shape = (3 * lapo_config['frame_stack'], dataset.img_hw, dataset.img_hw)
    if args.model == 'laom':
        is_deep = lapo_config.get('encoder_deep', False)
        scale = lapo_config.get('encoder_scale', 1)
        encoder_channels = (16, 32, 64, 128, 256) if is_deep else (16, 32, 32)

        model_kwargs = {
            'shape': input_shape,
            'true_act_dim': dataset.act_dim,
            'latent_act_dim': lapo_config['latent_action_dim'],
            'act_head_dim': lapo_config['act_head_dim'],
            'act_head_dropout': lapo_config.get('act_head_dropout', 0.0),
            'obs_head_dim': lapo_config['obs_head_dim'],
            'obs_head_dropout': lapo_config.get('obs_head_dropout', 0.0),
            'encoder_scale': scale,
            'encoder_channels': encoder_channels,
            'encoder_num_res_blocks': lapo_config['encoder_num_res_blocks'],
            'encoder_dropout': lapo_config.get('encoder_dropout', 0.0),
            'encoder_norm_out': lapo_config.get('encoder_norm_out', True),
        }

        model = LAOMWithLabels(**model_kwargs).to(DEVICE)
    elif args.model == 'actor':
        # BC 설정 우선 사용, 없으면 lapo 설정으로 대체
        is_deep = bc_config.get('encoder_deep', lapo_config.get('encoder_deep', False))
        scale = bc_config.get('encoder_scale', lapo_config.get('encoder_scale', 1))
        encoder_channels = (16, 32, 64, 128, 256) if is_deep else (16, 32, 32)
        encoder_num_res_blocks = bc_config.get('encoder_num_res_blocks', lapo_config.get('encoder_num_res_blocks', 1))
        dropout = bc_config.get('dropout', 0.0)

        model = Actor(
            shape=input_shape,
            num_actions=lapo_config['latent_action_dim'],
            encoder_scale=scale,
            encoder_channels=encoder_channels,
            encoder_num_res_blocks=encoder_num_res_blocks,
            dropout=dropout,
        ).to(DEVICE)
    else:
        raise ValueError(f"알 수 없는 모델 타입: {args.model}")

    load_checkpoint(model, args.checkpoint_path)
    model.eval()

    print("\n--- 4. 데이터 포인트 샘플링 및 잠재 행동 추출 ---")
    
    # 먼저 모든 가능한 (traj_idx, t, view_key) 조합을 생성
    all_combinations = []
    for traj_idx in range(num_traj_to_process):
        traj_len = dataset.traj_lens[traj_idx]
        loop_range = range(traj_len - args.future_offset)  # future_offset 사용
        
        for t in loop_range:
            for view_key in dataset.view_keys:
                all_combinations.append((traj_idx, t, view_key))
    
    print(f"총 {len(all_combinations)}개의 가능한 데이터 포인트 발견")
    
    # num_visualize만큼 랜덤 샘플링
    total_samples = len(all_combinations)
    num_to_visualize = min(args.num_visualize, total_samples)
    
    if num_to_visualize < total_samples:
        print(f"랜덤 샘플링: {total_samples}개 중 {num_to_visualize}개 선택")
        np.random.seed(42)  # 재현 가능한 샘플링
        sample_indices = np.random.choice(total_samples, num_to_visualize, replace=False)
        sampled_combinations = [all_combinations[i] for i in sample_indices]
    else:
        print(f"모든 {total_samples}개 데이터 포인트 사용")
        sampled_combinations = all_combinations
    
    # 샘플링된 조합들에 대해서만 latent action 추출
    latent_actions = []
    true_actions = []
    view_labels = []
    traj_indices = []
    timesteps = []
    
    for traj_idx, t, view_key in tqdm(sampled_combinations, desc="Extracting latent actions"):
        action = dataset.actions[traj_idx][t]
        obs = dataset._get_single_view_padded_obs(traj_idx, t, view_key)
        future_obs = dataset._get_single_view_padded_obs(traj_idx, t + args.future_offset, view_key)

        obs = obs.unsqueeze(0).to(DEVICE).permute(0, 3, 1, 2)
        future_obs = future_obs.unsqueeze(0).to(DEVICE).permute(0, 3, 1, 2)
        
        obs = normalize_img(obs)
        future_obs = normalize_img(future_obs)
        
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            if args.model == 'laom':
                _, latent_action, _ = model(obs, future_obs)
            else:
                # actor: 단일 obs만 사용
                act_logits, _ = model(obs)
                latent_action = act_logits
        
        latent_actions.append(latent_action.squeeze().cpu().float().numpy())
        true_actions.append(action)
        view_labels.append(view_key)
        traj_indices.append(traj_idx)
        timesteps.append(t)

    print(f"시각화할 데이터 포인트: {len(latent_actions)}개")
    # 샘플링 후 latent action 배열 shape 및 값 범위 출력
    latent_actions_array_tmp = np.array(latent_actions)
    print(f"latent_actions array shape: {latent_actions_array_tmp.shape}, value range: [{latent_actions_array_tmp.min():.4f}, {latent_actions_array_tmp.max():.4f}]")

    # checkpoint 경로에서 epoch 추출
    checkpoint_name = os.path.basename(args.checkpoint_path)
    if 'epoch_' in checkpoint_name:
        epoch = checkpoint_name.split('epoch_')[1].split('.')[0]
    else:
        epoch = "unknown"
    
    # 출력 디렉토리 생성 (checkpoint 경로 기준)
    ckpt_dir = os.path.dirname(args.checkpoint_path)
    output_dir = os.path.join(ckpt_dir, "umap_visualization")
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성
    base_filename = f"offset_{args.future_offset}_epoch_{epoch}"

    print("\n--- 5. 실제 행동 군집화 ---")
    latent_actions_array = latent_actions_array_tmp
    true_actions_array = np.array(true_actions)

    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    action_bins = kmeans.fit_predict(true_actions_array)
    print(f"{n_clusters}개의 행동 bin으로 군집화 완료.")

    print("\n--- 6. UMAP 차원 축소 및 시각화 ---")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(latent_actions_array)
    
    df = pd.DataFrame({
        'umap1': embedding[:, 0],
        'umap2': embedding[:, 1],
        'view': view_labels,
        'action_bin': [f"Bin {b}" for b in action_bins]
    })

    # --- 추가: 7 DoF 액션 각 차원을 이진화하여 서브플롯 시각화 ---
    if true_actions_array.ndim == 2 and true_actions_array.shape[1] >= 7:
        per_dim_thresholds = np.median(true_actions_array, axis=0)
        for dim_idx in range(7):
            df[f'a{dim_idx}_bin'] = (true_actions_array[:, dim_idx] >= per_dim_thresholds[dim_idx]).astype(int)

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes = axes.flatten()
        binary_palette = sns.color_palette(["#1f77b4", "#ff7f0e"])  # 0/1
        for dim_idx in range(7):
            ax = axes[dim_idx]
            sns.scatterplot(
                data=df,
                x='umap1',
                y='umap2',
                hue=f'a{dim_idx}_bin',
                palette=binary_palette,
                s=40,
                alpha=0.7,
                legend=False,
                ax=ax
            )
            ax.set_title(f'Dim {dim_idx} (thr={per_dim_thresholds[dim_idx]:.3f})')
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.grid(True)
        # 마지막 빈 축 정리
        axes[7].axis('off')
        plt.tight_layout()
        per_dim_path = os.path.join(output_dir, f"{base_filename}_per_dim_binary.png")
        plt.savefig(per_dim_path)
        print(f"차원별 이진화 서브플롯을 '{per_dim_path}'에 저장했습니다.")
        plt.close()
    else:
        print("경고: true_actions_array의 차원이 7 미만이므로 차원별 이진화 서브플롯을 건너뜁니다.")
    # --- 추가 끝 ---

    # output_dir은 이미 위에서 생성됨

    # 행동 bin에 대한 색상 팔레트 생성
    unique_bins = sorted(list(df['action_bin'].unique()))
    action_palette = sns.color_palette("husl", n_colors=len(unique_bins))
    
    # 뷰에 대한 마커(모양) 생성
    unique_views = sorted(list(df['view'].unique()))
    # Seaborn에서 혼합할 수 없는 '+' 와 'x' 같은 선 마커를 제거하고 채워진 마커만 사용합니다.
    markers_list = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'h', 'H', 'X', '8']
    view_markers = {view: markers_list[i % len(markers_list)] for i, view in enumerate(unique_views)}

    # 1. 모든 뷰와 액션 bin을 함께 시각화 (색상=액션, 모양=뷰)
    print("Generating combined plot for all views...")
    plt.figure(figsize=(16, 14))
    sns.scatterplot(
        data=df,
        x='umap1',
        y='umap2',
        hue='action_bin',
        style='view',
        hue_order=unique_bins,
        style_order=unique_views,
        palette=action_palette,
        markers=view_markers,
        s=80,
        alpha=0.7
    )
    
    plt.title(f'UMAP: Color=Action Bin, Shape=View ({len(latent_actions)} samples from {num_traj_to_process} trajectories)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    all_views_path = os.path.join(output_dir, f"{base_filename}_all_views_combined.png")
    plt.savefig(all_views_path)
    print(f"전체 통합 시각화 결과를 '{all_views_path}'에 저장했습니다.")
    plt.close()

    # 2. 각 뷰를 개별적으로 시각화 (해당 뷰의 모양 + 액션 bin의 색상)
    for view_to_highlight in tqdm(unique_views, desc="Generating plots for each view"):
        plt.figure(figsize=(12, 10))
        
        highlight_df = df[df['view'] == view_to_highlight]
        sns.scatterplot(
            data=highlight_df, 
            x='umap1', 
            y='umap2',
            hue='action_bin',
            hue_order=unique_bins,
            palette=action_palette,
            style='view', # 단일 모양을 지정하기 위해 사용
            style_order=[view_to_highlight],
            markers=view_markers,
            s=120,
            alpha=0.9,
            legend='full'
        )
        
        leg = plt.gca().get_legend()
        if leg:
            leg.set_title("Action Bins")

        plt.title(f'UMAP for {view_to_highlight} ({len(latent_actions)} samples from {num_traj_to_process} trajectories)')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.grid(True)
        plt.tight_layout()
        
        view_path = os.path.join(output_dir, f"{base_filename}_{view_to_highlight.replace('/', '_')}.png")
        plt.savefig(view_path)
        plt.close()
    
    print(f"각 뷰별 개별 시각화 결과를 '{output_dir}/{base_filename}_*.png' 패턴으로 저장했습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMAP으로 잠재 행동 시각화 (LAOM 또는 Actor).")
    parser.add_argument('--model', type=str, choices=['laom', 'actor'], default='laom',
                        help='사용할 모델 타입: laom 또는 actor (default: laom)')
    
    parser.add_argument('--checkpoint-path', type=str, required=True, 
                        help='Path to the LAOM model checkpoint (.pt file)')
    parser.add_argument('--data-path', type=str, required=True, 
                        help='Path to the HDF5 dataset')
    # parser.add_argument('--config-path', type=str, required=True,
    #                     help='Path to the config YAML file used for training.')
    
    parser.add_argument('--output-prefix', type=str, default='umap_visualization', 
                        help='Prefix for the output plot files (default: umap_visualization)')
    parser.add_argument('--num-visualize', type=int, default=10000, 
                        help='Number of samples to visualize after loading all trajectories (default: 10000)')
    parser.add_argument('--future-offset', type=int, default=1, 
                        help='Offset for future observation (default: 1)')
    parser.add_argument('--views', type=str, default='all',
                        help='Views to visualize: "all" for all views (0-20), or comma-separated indices like "0,1,2" (default: all)')

    args = parser.parse_args()
    visualize(args)

import gym
import numpy as np
import torch
import env
from env.venv import SubprocVectorEnv
import os
import yaml
from dataclasses import dataclass, field
import gym.wrappers
from src.nn import Actor
from src.utils import normalize_img
from tqdm import trange
import torch.nn as nn
import imageio
from datetime import datetime
import h5py
from skimage.transform import resize


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class OnlyDecoderConfig:
    num_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 2
    encoder_deep: bool = False
    dropout: float = 0.0
    use_aug: bool = True
    frame_stack: int = 3
    data_path: str = "/shared/s2/lab01/dataset/DMC/pusht_view1.h5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 5
    eval_seed: int = 0
    offset: int = 5


def get_dataset_attrs(hdf5_path):
    action_mean, action_std, img_hw = None, None, None
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, "r") as df:
            if "action_mean" in df.attrs:
                action_mean = df.attrs["action_mean"]
            if "action_std" in df.attrs:
                action_std = df.attrs["action_std"]
            if "img_hw" in df.attrs:
                val = df.attrs["img_hw"]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 2:
                    img_hw = tuple(val)
                elif isinstance(val, (int, np.integer)):
                    img_hw = (int(val), int(val))
                else:
                    print(f"Warning: 'img_hw' attribute in {hdf5_path} has unexpected format: {val}. Using default resolution.")
                    img_hw = None
    else:
        print(f"Warning: Dataset file not found at {hdf5_path}, cannot get dataset attributes.")
    return action_mean, action_std, img_hw


def load_checkpoint(model, filepath):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)

        # Handle potential DataParallel wrapper
        state_dict = checkpoint['model_state_dict']
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # strict=False로 로드하여 shape mismatch를 허용
        try:
            model.load_state_dict(state_dict)
            print(f"체크포인트 불러옴: {filepath} (epoch {checkpoint.get('epoch', 'N/A')})")
        except RuntimeError as e:
            print(f"경고: 일부 파라미터가 로드되지 않았습니다: {e}")
            # strict=False로 다시 시도
            model.load_state_dict(state_dict, strict=False)
            print(f"체크포인트 일부 불러옴: {filepath}")

        config_filepath = filepath.replace('.pt', '_config.yaml')
        if os.path.exists(config_filepath):
            with open(config_filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
            print(f"Config 파일 불러옴: {config_filepath}")
            return config_dict
        return None
    else:
        print(f"경고: 체크포인트 파일을 찾을 수 없습니다: {filepath}")
        return None


class VisualEnvWrapper(gym.Wrapper):
    def __init__(self, env, height, width):
        super().__init__(env)
        self.height = height
        self.width = width
        
        # 원본 observation_space
        original_obs_space = env.observation_space
        
        # observation_space가 Dict 형태이고 'visual' 키를 포함하는지 확인
        if hasattr(original_obs_space, 'spaces') and 'visual' in original_obs_space.spaces:
            visual_space = original_obs_space['visual']
        else:
            visual_space = original_obs_space

        # 새로운 observation_space 정의
        old_shape = visual_space.shape
        new_shape = (height, width, old_shape[2])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def _process_obs(self, obs):
        # observation이 dict인 경우 visual 키 사용
        if isinstance(obs, dict) and 'visual' in obs:
            obs = obs['visual']
        
        # 이미지 리사이징
        return resize(obs, (self.height, self.width), preserve_range=True).astype(np.uint8)

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        return self._process_obs(obs_dict), reward, done, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, _ = result
        else:
            obs = result
        return self._process_obs(obs)

#2. create the environment
cfg = OnlyDecoderConfig()

action_mean, action_std, img_hw = get_dataset_attrs(cfg.data_path)
if action_mean is not None and action_std is not None:
    action_mean = torch.tensor(action_mean, device=DEVICE, dtype=torch.float32)
    action_std = torch.tensor(action_std, device=DEVICE, dtype=torch.float32)
    print("Action stats loaded for un-normalization.")

img_h, img_w = (224, 224)
if img_hw is not None:
    img_h, img_w = img_hw
    print(f"Using image size from dataset: {img_h}x{img_w}")

env_name = "pusht"
num_eval_envs = 5
env_kwargs = {
    "with_velocity": True,
    "with_target": True
}

def make_wrapped_env(height, width):
    def _init():
        env = gym.make(env_name, **env_kwargs)
        env = VisualEnvWrapper(env, height, width)
        env = gym.wrappers.FrameStack(env, num_stack=cfg.frame_stack)
        return env
    return _init

env = SubprocVectorEnv(
    [make_wrapped_env(img_h, img_w) for _ in range(num_eval_envs)]
)

#3. create the model
# SubprocVectorEnv에서는 observation_space가 리스트로 반환되므로 첫 번째 것을 사용
obs_space = env.observation_space[0] if isinstance(env.observation_space, list) else env.observation_space
act_space = env.action_space[0] if isinstance(env.action_space, list) else env.action_space

obs_shape = obs_space.shape
act_dim = act_space.shape[0]

print(f"Observation shape: {obs_shape}")
print(f"Action dimension: {act_dim}")

# FrameStack 적용 후 shape는 (frame_stack, H, W, C) 형태
# Actor의 shape 파라미터는 (channels, height, width) 형태로 전달
if len(obs_shape) == 4:  # (frame_stack, H, W, C)
    frame_stack, height, width, channels = obs_shape
    channels = channels * frame_stack  # 프레임 스태킹으로 인해 채널 수 증가
elif len(obs_shape) == 3:  # (H, W, C) 형태
    channels, height, width = obs_shape[2], obs_shape[0], obs_shape[1]
else:
    print(f"Unexpected observation shape: {obs_shape}")
    # 기본값 사용
    channels, height, width = 9, 224, 224

checkpoint_path = "/shared/s2/lab01/youngjoonjeong/LAOM/checkpoints/2025-08-27_12-40-44/only_decoder_final.pt"
config_from_ckpt = None
config_filepath = checkpoint_path.replace('.pt', '_config.yaml')
if os.path.exists(config_filepath):
    with open(config_filepath, 'r') as f:
        config_from_ckpt = yaml.safe_load(f)
    print(f"Config file loaded: {config_filepath}")

train_cfg = cfg
if config_from_ckpt and 'only_decoder' in config_from_ckpt:
    train_cfg_dict = config_from_ckpt['only_decoder']
    # Filter only keys that are present in the OnlyDecoderConfig
    known_keys = {f.name for f in field(OnlyDecoderConfig).values()}
    filtered_dict = {k: v for k, v in train_cfg_dict.items() if k in known_keys}
    train_cfg = OnlyDecoderConfig(**filtered_dict)
    print("Using config from checkpoint.")
else:
    print("Using default config for evaluation.")

final_act_dim = act_dim * train_cfg.offset
print(f"Action dimension: {act_dim}, Offset: {train_cfg.offset}, Final action dim: {final_act_dim}")

actor = Actor(
    shape=(channels, height, width),
    num_actions=final_act_dim,
    encoder_scale=train_cfg.encoder_scale,
    encoder_channels=(16, 32, 32) if not train_cfg.encoder_deep else (16, 32, 64, 128, 256),
    encoder_num_res_blocks=train_cfg.encoder_num_res_blocks,
    dropout=train_cfg.dropout,
).to(DEVICE)

load_checkpoint(actor, checkpoint_path)
actor.eval()


#5. evaluate the model
@torch.no_grad()
def evaluate_policy(env, actor_model, num_episodes, real_act_dim, seed=0, device="cpu", max_steps=1000, video_dir=None, action_mean=None, action_std=None):
    """
    num_episodes는 평가 배치 수를 의미합니다.
    총 에피소드 수 = num_episodes * num_envs
    """
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
        print(f"평가 영상이 {video_dir}에 저장됩니다.")

    all_returns = []
    all_successes = []
    num_envs = env.env_num
    
    for ep_batch in trange(num_episodes, desc="Evaluating Batches", leave=False):
        batch_seeds = [seed + ep_batch * num_envs + i for i in range(num_envs)]
        env.seed(batch_seeds)
        obs = env.reset()
        
        dones = np.zeros(num_envs, dtype=bool)
        episode_rewards = np.zeros(num_envs, dtype=np.float32)
        max_coverages = np.zeros(num_envs, dtype=np.float32)
        frames = [[] for _ in range(num_envs)]
        step = 0

        while not np.all(dones) and step < max_steps:
            if video_dir:
                for i in range(num_envs):
                    if not dones[i]:
                        # obs shape: (num_envs, frame_stack, H, W, C), 마지막 프레임(RGB)을 저장합니다.
                        frames[i].append(np.array(obs[i][-1], dtype=np.uint8))

            obs_np = np.array(obs)
            obs_tensor = torch.tensor(obs_np, device=device)
            # FrameStack 적용 후 shape: (num_envs, frame_stack, H, W, C)
            # 모델 입력 shape: (num_envs, frame_stack * C, H, W)
            if len(obs_tensor.shape) == 5:  # (num_envs, frame_stack, H, W, C)
                obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3)  # (num_envs, frame_stack, C, H, W)
                obs_tensor = obs_tensor.reshape(obs_tensor.shape[0], -1, obs_tensor.shape[3], obs_tensor.shape[4])  # (num_envs, frame_stack * C, H, W)
            obs_tensor = normalize_img(obs_tensor)

            action_tensor, _ = actor_model(obs_tensor)
            action_tensor = action_tensor[:, :real_act_dim]

            if action_mean is not None and action_std is not None:
                action_tensor = action_tensor * action_std + action_mean
            
            action_np = action_tensor.cpu().numpy()
            action_np = action_np / 100
            print("INFO: action_np: ", action_np)

            next_obs, rewards, dones, infos = env.step(action_np)
            # print("INFO: next_obs: ", next_obs, type(next_obs))
            # print("INFO: rewards: ", rewards, type(rewards))
            # print("INFO: dones: ", dones, type(dones))
            # print("INFO: infos: ", infos, type(infos))

            episode_rewards += rewards * ~dones
            current_max_coverages = np.array([info.get('max_coverage', 0) for info in infos])
            max_coverages = np.maximum(max_coverages, current_max_coverages)
            obs = next_obs
            step += 1
        
        all_returns.extend(episode_rewards.tolist())
        successes = dones == True
        all_successes.extend(successes.tolist())
        

        if video_dir:
            for i in range(num_envs):
                video_path = os.path.join(video_dir, f"eval_batch_{ep_batch}_env_{i}_success_{successes[i]}.mp4")
                imageio.mimsave(video_path, frames[i], fps=10)

    return np.array(all_returns), np.mean(all_successes)

# 평가 결과 및 비디오 저장을 위한 디렉토리 생성
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
eval_dir = f"eval_output/{timestamp}"
os.makedirs(eval_dir, exist_ok=True)

returns, success_rate = evaluate_policy(
    env, actor, 
    num_episodes=cfg.eval_episodes // num_eval_envs, 
    real_act_dim=act_dim,
    seed=cfg.eval_seed, 
    device=DEVICE, 
    video_dir=eval_dir,
    action_mean=action_mean,
    action_std=action_std
)

# 결과 저장 및 출력
results = {
    "returns": returns.tolist(),
    "mean_return": np.mean(returns),
    "std_return": np.std(returns),
    "success_rate": success_rate,
}
with open(os.path.join(eval_dir, "results.yaml"), "w") as f:
    yaml.dump(results, f)

print(f"Evaluation results saved to {eval_dir}/results.yaml")
print(f"Mean return: {results['mean_return']:.2f}, Std return: {results['std_return']:.2f}")
print(f"Success rate: {results['success_rate']:.2f}")

env.close()
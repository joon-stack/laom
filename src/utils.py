import os
import random

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
from shimmy import DmControlCompatibilityV0
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from .dcs import suite


def set_seed(seed, env=None, deterministic_torch=False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def get_optim_groups(model, weight_decay):
    return [
        # do not decay biases and single-column parameters (rmsnorm), those are usually scales
        {"params": (p for p in model.parameters() if p.dim() < 2), "weight_decay": 0.0},
        {"params": (p for p in model.parameters() if p.dim() >= 2), "weight_decay": weight_decay},
    ]


def get_grad_norm(model):
    grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
    norm = torch.cat(grads).norm()
    return norm


def soft_update(target, source, tau=1e-3):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class DCSInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu"):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]

            if "action_mean" in df.attrs:
                self.action_mean = df.attrs["action_mean"]
                print("INFO: self.action_mean", self.action_mean)
            else:
                self.action_mean = None
            if "action_std" in df.attrs:
                self.action_std = df.attrs["action_std"]
                print("INFO: self.action_std", self.action_std)
            else:
                self.action_std = None
            self.new_actions = []
            if self.action_mean is not None and self.action_std is not None:
                for act in self.actions:
                    act = (act - self.action_mean) / self.action_std
                    self.new_actions.append(act)
                self.actions = self.new_actions

        self.frame_stack = frame_stack
        # 각 trajectory의 길이를 개별적으로 저장
        self.traj_lens = [obs.shape[0] for obs in self.observations]
        
        # 누적 길이를 미리 계산하여 저장
        self.cumulative_transitions = []
        total = 0
        for traj_len in self.traj_lens:
            valid_transitions = max(0, traj_len - 1)
            total += valid_transitions
            self.cumulative_transitions.append(total)

    def __get_padded_obs(self, traj_idx, idx):
        # 해당 trajectory의 실제 길이 사용
        traj_len = self.traj_lens[traj_idx]
        
        # 인덱스가 trajectory 길이를 초과하지 않도록 제한
        idx = min(idx, traj_len - 1)
        
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = min(idx + 1, traj_len)
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs

    def __len__(self):
        return self.cumulative_transitions[-1] if self.cumulative_transitions else 0

    def __getitem__(self, idx):
        # 누적 길이 배열에서 trajectory 찾기
        traj_idx = 0
        for i, cumulative in enumerate(self.cumulative_transitions):
            if idx < cumulative:
                traj_idx = i
                break
        
        # 해당 trajectory 내에서의 transition 인덱스 계산
        if traj_idx == 0:
            transition_idx = idx
        else:
            transition_idx = idx - self.cumulative_transitions[traj_idx - 1]

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
        action = self.actions[traj_idx][transition_idx]

        return obs, next_obs, action

class DCSLAPOInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.states = [torch.tensor(df[traj]["states"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]
            self.state_dim = self.states[0][0].shape[-1]

            # if "action_mean" in df.attrs:
            #     self.action_mean = df.attrs["action_mean"]
            #     print("INFO: self.action_mean", self.action_mean)
            # else:
            #     self.action_mean = None
            # if "action_std" in df.attrs:
            #     self.action_std = df.attrs["action_std"]
            #     print("INFO: self.action_std", self.action_std)
            # else:
            #     self.action_std = None

            self.action_mean = None
            self.action_std = None
            self.new_actions = []
            if self.action_mean is not None and self.action_std is not None:
                for act in self.actions:
                    act = (act - self.action_mean) / self.action_std
                    self.new_actions.append(act)
                self.actions = self.new_actions

        self.frame_stack = frame_stack
        # 각 trajectory의 길이를 개별적으로 저장
        self.traj_lens = [obs.shape[0] for obs in self.observations]
        self.max_offset = max_offset

        
        
        # 누적 길이를 미리 계산하여 저장
        self.cumulative_transitions = []
        total = 0
        for traj_len in self.traj_lens:
            valid_transitions = max(0, traj_len - self.max_offset)
            total += valid_transitions
            self.cumulative_transitions.append(total)
        
        # 최소 길이를 기준으로 max_offset 설정
        min_traj_len = min(self.traj_lens)
        assert 1 <= max_offset < min_traj_len
        self.max_offset = max_offset

    def __get_padded_obs(self, traj_idx, idx):
        # 해당 trajectory의 실제 길이 사용
        traj_len = self.traj_lens[traj_idx]
        
        # 인덱스가 trajectory 길이를 초과하지 않도록 제한
        idx = min(idx, traj_len - 1)
        
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = min(idx + 1, traj_len)
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs

    def __len__(self):
        return self.cumulative_transitions[-1] if self.cumulative_transitions else 0

    def __getitem__(self, idx):
        # 누적 길이 배열에서 trajectory 찾기
        traj_idx = 0
        for i, cumulative in enumerate(self.cumulative_transitions):
            if idx < cumulative:
                traj_idx = i
                break
        
        # 해당 trajectory 내에서의 transition 인덱스 계산
        if traj_idx == 0:
            transition_idx = idx
        else:
            transition_idx = idx - self.cumulative_transitions[traj_idx - 1]

        action = self.actions[traj_idx][transition_idx]
        state = self.states[traj_idx][transition_idx]

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
        offset = random.randint(1, min(self.max_offset, self.traj_lens[traj_idx] - transition_idx - 1))
        future_obs = self.__get_padded_obs(traj_idx, transition_idx + offset)

        return obs, next_obs, future_obs, action, state

class DCSLAOMInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.states = [torch.tensor(df[traj]["states"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]
            self.state_dim = self.states[0][0].shape[-1]

            # if "action_mean" in df.attrs:
            #     self.action_mean = df.attrs["action_mean"]
            #     print("INFO: self.action_mean", self.action_mean)
            # else:
            #     self.action_mean = None
            # if "action_std" in df.attrs:
            #     self.action_std = df.attrs["action_std"]
            #     print("INFO: self.action_std", self.action_std)
            # else:
            #     self.action_std = None
            self.action_mean = None
            self.action_std = None

            self.new_actions = []
            if self.action_mean is not None and self.action_std is not None:
                for act in self.actions:
                    act = (act - self.action_mean) / self.action_std
                    self.new_actions.append(act)
                self.actions = self.new_actions

        self.frame_stack = frame_stack
        # 각 trajectory의 길이를 개별적으로 저장
        self.traj_lens = [obs.shape[0] for obs in self.observations]

        
        # 최소 길이를 기준으로 max_offset 설정
        min_traj_len = min(self.traj_lens)
        assert 1 <= max_offset < min_traj_len
        self.max_offset = max_offset
        
        # 누적 길이를 미리 계산하여 저장
        self.cumulative_transitions = []
        total = 0
        for traj_len in self.traj_lens:
            valid_transitions = max(0, traj_len - self.max_offset)
            total += valid_transitions
            self.cumulative_transitions.append(total)

    def __get_padded_obs(self, traj_idx, idx):
        # 해당 trajectory의 실제 길이 사용
        traj_len = self.traj_lens[traj_idx]
        
        # 인덱스가 trajectory 길이를 초과하지 않도록 제한
        idx = min(idx, traj_len - 1)
        
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = min(idx + 1, traj_len)
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs

    def __len__(self):
        return self.cumulative_transitions[-1] if self.cumulative_transitions else 0

    def __getitem__(self, idx):
        # 누적 길이 배열에서 trajectory 찾기
        traj_idx = 0
        for i, cumulative in enumerate(self.cumulative_transitions):
            if idx < cumulative:
                traj_idx = i
                break
        
        # 해당 trajectory 내에서의 transition 인덱스 계산
        if traj_idx == 0:
            transition_idx = idx
        else:
            transition_idx = idx - self.cumulative_transitions[traj_idx - 1]

        action = self.actions[traj_idx][transition_idx]
        state = self.states[traj_idx][transition_idx]

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
        offset = random.randint(1, min(self.max_offset, self.traj_lens[traj_idx] - transition_idx - 1))
        future_obs = self.__get_padded_obs(traj_idx, transition_idx + offset)

        return obs, next_obs, future_obs, action, state, (offset - 1)


class DCSMVInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1, camera_num=8, view_keys=None, resize_wh=(128, 128), view_keys_to_load=None, num_trajectories_to_load=None, selected_demo_names=None, mixed_view_sampling=False, positive_samples_per_instance=4):
        """
        Multi-view DCS dataset for LAOM training with instance group sampling.
        Based on DCSLAOMInMemoryDataset with multi-view support.
        
        Args:
            hdf5_path: Path to HDF5 file
            frame_stack: Number of frames to stack
            device: Device to load data on
            max_offset: Maximum offset for future observations
            camera_num: Number of camera views to sample per batch (used in collate_fn)
            view_keys: List of view keys to use (e.g., ["view_00/agentview_image", "view_01/agentview_image"])
            resize_wh: Image resize dimensions
        """
        self.frame_stack = frame_stack
        self.device = device
        self.max_offset = max_offset
        self.camera_num = camera_num
        self.view_keys = view_keys or []
        self.resize_wh = resize_wh
        self.mixed_view_sampling = mixed_view_sampling
        self.positive_samples_per_instance = positive_samples_per_instance
        
        # Determine which keys to load from HDF5
        self.view_keys_to_load = view_keys_to_load if view_keys_to_load is not None else self.view_keys
        # Pre-selected demo names (trajectory IDs) to restrict loading
        self.selected_demo_names = set(selected_demo_names) if selected_demo_names is not None else None

        # Load multi-view data, preserving trajectory structure
        self.data_dict = {view: [] for view in self.view_keys_to_load}
        self.load_multi_view_data(hdf5_path, num_trajectories_to_load)
        
        # Load actions and states, preserving trajectory structure
        self.actions = []
        self.states = []
        self.load_actions_states(hdf5_path, num_trajectories_to_load)
        
        # Calculate trajectory lengths and cumulative transitions
        if self.view_keys_to_load and not self.data_dict[self.view_keys_to_load[0]]:
             raise ValueError("No trajectories loaded. Check your HDF5 file path and view keys.")
        self.traj_lens = [len(traj) for traj in self.data_dict[self.view_keys_to_load[0]]]

        min_traj_len = min(self.traj_lens) if self.traj_lens else 0
        if min_traj_len <= max_offset:
            print(f"Warning: max_offset ({max_offset}) is not smaller than the minimum trajectory length ({min_traj_len}). Adjusting max_offset.")
            self.max_offset = min_traj_len -1
        else:
            self.max_offset = max_offset
        
        # Set img_hw from resize_wh
        self.img_hw = self.resize_wh[0]  # Assuming square images
        
        # Calculate total valid moments (transitions) across all trajectories
        self.total_moments = 0
        for traj_len in self.traj_lens:
            self.total_moments += max(0, traj_len - self.max_offset)
        
        # Create mapping from moment index to (traj_idx, transition_idx)
        self.moment_to_traj_transition = []
        for traj_idx, traj_len in enumerate(self.traj_lens):
            valid_transitions = max(0, traj_len - self.max_offset)
            for transition_idx in range(valid_transitions):
                self.moment_to_traj_transition.append((traj_idx, transition_idx))

        self.fixed_offset = None

    def set_fixed_offset(self, offset):
        """Set a fixed offset for all subsequent __getitem__ calls."""
        self.fixed_offset = offset

    def load_multi_view_data(self, hdf5_path, num_trajectories_to_load=None):
        """Load multi-view image data from HDF5 file, preserving trajectory structure."""
        print(f"Loading multi-view data from HDF5 file: {hdf5_path}...")
        try:
            with h5py.File(hdf5_path, 'r') as f:
                demos = sorted(list(f['data'].keys()))
                # Filter demos by pre-selected names if provided
                if self.selected_demo_names is not None:
                    demos = [d for d in demos if d in self.selected_demo_names]
                if num_trajectories_to_load is not None:
                    demos = demos[:num_trajectories_to_load]

                # Persist the demo order used in this dataset instance
                self.demo_names = list(demos)

                for demo in tqdm(demos, desc="Loading Multi-view Demos"):
                    temp_traj_data = {view: [] for view in self.view_keys_to_load}
                    
                    # First, check if all required views exist for this demo
                    is_demo_valid = True
                    for view_key in self.view_keys_to_load:
                        full_key = f'data/{demo}/obs/{view_key}'
                        if full_key not in f:
                            print(f"Warning: View {view_key} not found in demo {demo}. Skipping this demo.")
                            is_demo_valid = False
                            break
                    if not is_demo_valid:
                        continue

                    # Load images for the valid demo
                    for view_key in self.view_keys_to_load:
                        images = f[f'data/{demo}/obs/{view_key}'][()]
                        resized_images = [np.array(Image.fromarray(img).resize(self.resize_wh)) for img in images]
                        temp_traj_data[view_key] = np.stack(resized_images)
                    
                    # Append trajectory data for all views
                    for view_key in self.view_keys_to_load:
                        self.data_dict[view_key].append(temp_traj_data[view_key])

        except Exception as e:
            print(f"Error loading HDF5 file: {e}")
            raise
        
        print("Multi-view data loaded successfully.")

    def load_actions_states(self, hdf5_path, num_trajectories_to_load=None):
        """Load actions and states from HDF5 file, preserving trajectory structure."""
        print("Loading actions and states...")
        try:
            with h5py.File(hdf5_path, 'r') as f:
                demos = sorted(list(f['data'].keys()))
                # Apply the same demo filtering
                if self.selected_demo_names is not None:
                    demos = [d for d in demos if d in self.selected_demo_names]
                if num_trajectories_to_load is not None:
                    demos = demos[:num_trajectories_to_load]

                # Keep the same order with images
                if hasattr(self, 'demo_names'):
                    demos = [d for d in self.demo_names if d in set(demos)]

                for demo in tqdm(demos, desc="Loading Actions and States"):
                    # A simple check to ensure we only load actions/states for demos that had all views
                    is_demo_valid = all(f'data/{demo}/obs/{view_key}' in f for view_key in self.view_keys_to_load)
                    if not is_demo_valid:
                        continue

                    # Load actions
                    if f'data/{demo}/actions' in f:
                        actions = f[f'data/{demo}/actions'][()]
                        self.actions.append(actions)
                    
                    # Load states  
                    if f'data/{demo}/states' in f:
                        states = f[f'data/{demo}/states'][()]
                        self.states.append(states)
                
                # Get dimensions from the first trajectory
                if self.actions:
                    self.act_dim = self.actions[0].shape[-1]
                if self.states:
                    self.state_dim = self.states[0].shape[-1]
                
                # Load normalization parameters if available
                if "action_mean" in f.attrs:
                    self.action_mean = f.attrs["action_mean"]
                    print("INFO: self.action_mean", self.action_mean)
                else:
                    self.action_mean = None
                if "action_std" in f.attrs:
                    self.action_std = f.attrs["action_std"]
                    print("INFO: self.action_std", self.action_std)
                else:
                    self.action_std = None
                    
        except Exception as e:
            print(f"Error loading actions/states: {e}")
            raise

    def _get_single_view_padded_obs(self, traj_idx, idx, view_key):
        """Get single view observation with frame stacking and padding."""
        # Get observations for this view and trajectory
        traj_view_obs = self.data_dict[view_key][traj_idx]
        traj_len = len(traj_view_obs)
        
        # Handle empty trajectory
        if traj_len == 0:
            # Create dummy observation with proper shape (H, W, T*C)
            dummy_shape = (*self.resize_wh, self.frame_stack * 3)
            return torch.zeros(dummy_shape, device=self.device, dtype=torch.uint8)
        
        # Limit index to trajectory length
        idx = min(idx, traj_len - 1)
        
        # Frame stacking (same as original DCSLAOMInMemoryDataset)
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = min(idx + 1, traj_len)
        obs = traj_view_obs[min_obs_idx:max_obs_idx]

        # Pad if at the beginning (same as original)
        if len(obs) < self.frame_stack:
            pad_img = obs[0][None]
            obs = np.concatenate([pad_img] * (self.frame_stack - len(obs)) + [obs])
        
        # Convert to tensor and reshape: (T, H, W, C) -> (H, W, T*C)
        obs = torch.as_tensor(np.array(obs), device=self.device)
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)
        
        return obs

    def __len__(self):
        return self.total_moments

    def __getitem__(self, idx):
        # idx는 이제 '전체 데이터셋에서 몇 번째 순간인가'를 의미합니다.
        if idx >= len(self.moment_to_traj_transition):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.moment_to_traj_transition)} moments")
        
        # 1. idx를 (traj_idx, transition_idx)로 변환
        traj_idx, transition_idx = self.moment_to_traj_transition[idx]
        
        # 2. 해당 (traj_idx, transition_idx)에 대한 모든 뷰의 데이터를 가져옵니다.
        #    이것이 하나의 '인스턴스 그룹'이 됩니다.
        all_views_obs = []
        all_views_next_obs = []
        all_views_future_obs = []
        obs_camera_indices = []  # 새로 추가
        future_camera_indices = []  # 새로 추가
        
        # Calculate offset once per moment, not per view
        current_traj_len = self.traj_lens[traj_idx]
        max_possible_offset = current_traj_len - transition_idx - 1
        
        if self.fixed_offset is not None:
            # Ensure the fixed offset is valid for this transition
            offset = min(self.fixed_offset, max_possible_offset)
        else:
            offset = random.randint(1, min(self.max_offset, max_possible_offset))

        if self.mixed_view_sampling:
            # Mixed view sampling: generate multiple positive samples with different view combinations
            # Debug: Print for first few samples
            if idx < 2:
                print(f"[DEBUG Dataset] idx={idx}, mixed_view_sampling=True, generating {self.positive_samples_per_instance} samples")
            for sample_idx in range(self.positive_samples_per_instance):
                obs_view = random.choice(self.view_keys_to_load)
                next_obs_view = random.choice(self.view_keys_to_load)
                future_obs_view = random.choice(self.view_keys_to_load)
                
                if idx < 2:
                    print(f"  Sample {sample_idx}: obs_view={obs_view}, next_obs_view={next_obs_view}, future_obs_view={future_obs_view}")
                
                obs = self._get_single_view_padded_obs(traj_idx, transition_idx, obs_view)
                next_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + 1, next_obs_view)
                future_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, future_obs_view)
                
                all_views_obs.append(obs)
                all_views_next_obs.append(next_obs)
                all_views_future_obs.append(future_obs)
                
                # Camera indices 추가
                obs_camera_indices.append(self.view_keys_to_load.index(obs_view))
                future_camera_indices.append(self.view_keys_to_load.index(future_obs_view))
        else:
            # Original behavior: all observations from the same view
            for view_key in self.view_keys_to_load:
                obs = self._get_single_view_padded_obs(traj_idx, transition_idx, view_key)
                next_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + 1, view_key)
                future_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, view_key)
                
                all_views_obs.append(obs)
                all_views_next_obs.append(next_obs)
                all_views_future_obs.append(future_obs)
                
                # Camera indices 추가 (obs와 future가 같은 view이므로 동일)
                view_idx = self.view_keys_to_load.index(view_key)
                obs_camera_indices.append(view_idx)
                future_camera_indices.append(view_idx)

        # 3. 공통 정보 (액션, 상태, ID) 가져오기
        # 이 순간의 GT 액션은 모든 뷰에서 동일합니다.
        action = torch.tensor(self.actions[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
        state = torch.tensor(self.states[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
        
        # 액션 시퀀스 추가 (offset 만큼의 액션들을 concatenate)
        action_sequence = self.actions[traj_idx][transition_idx : transition_idx + offset]
        action_sequence = torch.tensor(action_sequence, device=self.device, dtype=torch.float32)
        
        # 이 '순간'의 고유 ID. (traj_idx, transition_idx) 조합을 기반으로 생성
        # 같은 trajectory+transition에서 생성된 모든 positive samples는 같은 instance_id를 가짐
        instance_id = traj_idx * 10000 + transition_idx  # 충분히 큰 multiplier 사용 

        return {
            "obs": torch.stack(all_views_obs), # (num_views, H, W, T*C)
            "next_obs": torch.stack(all_views_next_obs),
            "future_obs": torch.stack(all_views_future_obs),
            "action": action,
            "action_sequence": action_sequence,
            "state": state,
            "instance_id": instance_id,
            "offset": offset - 1,
            "obs_camera_idx": torch.tensor(obs_camera_indices, dtype=torch.long),  # 새로 추가
            "future_camera_idx": torch.tensor(future_camera_indices, dtype=torch.long),  # 새로 추가
        }


class DCSMVTrueActionsDataset(IterableDataset):
    _debug_counter = 0  # Class variable for debugging
    
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1, camera_num=8, view_keys=None, resize_wh=(128, 128), view_keys_to_load=None, num_trajectories_to_load=None, selected_demo_names=None, mixed_view_sampling=False, positive_samples_per_instance=4):
        """
        Multi-view DCS dataset for LAOM training with infinite random sampling.
        This version loads data trajectory by trajectory to prevent data mixing between trajectories.
        """
        self.frame_stack = frame_stack
        self.device = device
        self.max_offset = max_offset
        self.camera_num = camera_num
        self.view_keys = view_keys or []
        self.resize_wh = resize_wh
        self.mixed_view_sampling = mixed_view_sampling
        self.positive_samples_per_instance = positive_samples_per_instance
        
        self.view_keys_to_load = view_keys_to_load if view_keys_to_load is not None else self.view_keys
        # Pre-selected demo names to restrict random sampling space
        self.selected_demo_names = set(selected_demo_names) if selected_demo_names is not None else None
        
        self.data_dict = {view: [] for view in self.view_keys_to_load}
        self.load_multi_view_data(hdf5_path, num_trajectories_to_load)
        
        self.actions = []
        self.states = []
        self.load_actions_states(hdf5_path, num_trajectories_to_load)
        
        if not self.view_keys_to_load or not self.data_dict[self.view_keys_to_load[0]]:
            raise ValueError("No trajectories loaded for Labeled Dataset. Check HDF5 path and view keys.")
        
        self.traj_lens = [len(traj) for traj in self.data_dict[self.view_keys_to_load[0]]]
        
        min_traj_len = min(self.traj_lens) if self.traj_lens else 0
        if min_traj_len <= max_offset:
            print(f"Warning (Labeled Dataset): max_offset ({max_offset}) is not smaller than min_traj_len ({min_traj_len}). Adjusting.")
            self.max_offset = min_traj_len - 1
        else:
            self.max_offset = max_offset
            
        self.img_hw = self.resize_wh[0]
        self.fixed_offset = None

    def set_fixed_offset(self, offset):
        self.fixed_offset = offset

    def load_multi_view_data(self, hdf5_path, num_trajectories_to_load=None):
        DCSMVInMemoryDataset.load_multi_view_data(self, hdf5_path, num_trajectories_to_load)

    def load_actions_states(self, hdf5_path, num_trajectories_to_load=None):
        DCSMVInMemoryDataset.load_actions_states(self, hdf5_path, num_trajectories_to_load)

    def _get_single_view_padded_obs(self, traj_idx, idx, view_key):
        return DCSMVInMemoryDataset._get_single_view_padded_obs(self, traj_idx, idx, view_key)

    def __iter__(self):
        while True:
            # Sample a trajectory respecting selected_demo_names if provided
            if self.selected_demo_names is not None:
                # Build a mapping from demo name order to index once
                pass
            traj_idx = random.randint(0, len(self.traj_lens) - 1)
            traj_len = self.traj_lens[traj_idx]
            
            if traj_len <= self.max_offset:
                continue
            
            transition_idx = random.randint(0, traj_len - self.max_offset - 1)

            all_views_obs, all_views_next_obs, all_views_future_obs = [], [], []
            obs_camera_indices = []  # 새로 추가
            future_camera_indices = []  # 새로 추가
            
            max_possible_offset = traj_len - transition_idx - 1
            if self.fixed_offset is not None:
                offset = min(self.fixed_offset, max_possible_offset)
            else:
                offset = random.randint(1, min(self.max_offset, max_possible_offset))

            if self.mixed_view_sampling:
                # Mixed view sampling: generate multiple positive samples with different view combinations
                # Debug: Print for first sample
                if DCSMVTrueActionsDataset._debug_counter < 2:
                    print(f"[DEBUG TrueActionsDataset] _debug_counter={DCSMVTrueActionsDataset._debug_counter}, mixed_view_sampling=True, generating {self.positive_samples_per_instance} samples")
                for sample_idx in range(self.positive_samples_per_instance):
                    obs_view = random.choice(self.view_keys_to_load)
                    next_obs_view = random.choice(self.view_keys_to_load)
                    future_obs_view = random.choice(self.view_keys_to_load)
                    
                    if DCSMVTrueActionsDataset._debug_counter < 2:
                        print(f"  Sample {sample_idx}: obs_view={obs_view}, next_obs_view={next_obs_view}, future_obs_view={future_obs_view}")
                    
                    obs = self._get_single_view_padded_obs(traj_idx, transition_idx, obs_view)
                    next_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + 1, next_obs_view)
                    future_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, future_obs_view)
                    
                    all_views_obs.append(obs)
                    all_views_next_obs.append(next_obs)
                    all_views_future_obs.append(future_obs)
                    
                    # Camera indices 추가
                    obs_camera_indices.append(self.view_keys_to_load.index(obs_view))
                    future_camera_indices.append(self.view_keys_to_load.index(future_obs_view))
            else:
                # Original behavior: all observations from the same view
                for view_key in self.view_keys_to_load:
                    obs = self._get_single_view_padded_obs(traj_idx, transition_idx, view_key)
                    next_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + 1, view_key)
                    future_obs = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, view_key)
                    
                    all_views_obs.append(obs)
                    all_views_next_obs.append(next_obs)
                    all_views_future_obs.append(future_obs)
                    
                    # Camera indices 추가 (obs와 future가 같은 view이므로 동일)
                    view_idx = self.view_keys_to_load.index(view_key)
                    obs_camera_indices.append(view_idx)
                    future_camera_indices.append(view_idx)

            action = torch.tensor(self.actions[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
            state = torch.tensor(self.states[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
            
            action_sequence_np = self.actions[traj_idx][transition_idx : transition_idx + offset]
            action_sequence = torch.as_tensor(action_sequence_np, device=self.device, dtype=torch.float32).reshape(-1)

            # 같은 trajectory+transition에서 생성된 모든 positive samples는 같은 instance_id를 가짐
            instance_id = traj_idx * 10000 + transition_idx  # 충분히 큰 multiplier 사용

            DCSMVTrueActionsDataset._debug_counter += 1
            
            yield {
                "obs": torch.stack(all_views_obs),
                "next_obs": torch.stack(all_views_next_obs),
                "future_obs": torch.stack(all_views_future_obs),
                "action": action, # Corrected key name from "action"
                "action_sequence": action_sequence, # Corrected key name
                "state": state, # Corrected key name
                "instance_id": instance_id, # Corrected key name
                "offset": offset - 1, # Corrected key name
                "obs_camera_idx": torch.tensor(obs_camera_indices, dtype=torch.long),  # 새로 추가
                "future_camera_idx": torch.tensor(future_camera_indices, dtype=torch.long),  # 새로 추가
            }

class DCSViewActionInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1, camera_num=8, view_keys=None, resize_wh=(128, 128), view_keys_to_load=None, num_trajectories_to_load=None, selected_demo_names=None):
        """
        Multi-view DCS dataset for view-action pair learning.
        Generates view pairs for contrastive learning.
        
        Args:
            hdf5_path: Path to HDF5 file
            frame_stack: Number of frames to stack
            device: Device to load data on
            max_offset: Maximum offset for future observations
            camera_num: Number of camera views to sample per batch (used in collate_fn)
            view_keys: List of view keys to use (e.g., ["view_00/agentview_image", "view_01/agentview_image"])
            resize_wh: Image resize dimensions
        """
        self.frame_stack = frame_stack
        self.device = device
        self.max_offset = max_offset
        self.camera_num = camera_num
        self.view_keys = view_keys or []
        self.resize_wh = resize_wh
        
        # Determine which keys to load from HDF5
        self.view_keys_to_load = view_keys_to_load if view_keys_to_load is not None else self.view_keys
        # Pre-selected demo names (trajectory IDs) to restrict loading
        self.selected_demo_names = set(selected_demo_names) if selected_demo_names is not None else None

        # Load multi-view data, preserving trajectory structure
        self.data_dict = {view: [] for view in self.view_keys_to_load}
        self.load_multi_view_data(hdf5_path, num_trajectories_to_load)
        
        # Load actions and states, preserving trajectory structure
        self.actions = []
        self.states = []
        self.load_actions_states(hdf5_path, num_trajectories_to_load)
        
        # Calculate trajectory lengths and cumulative transitions
        if self.view_keys_to_load and not self.data_dict[self.view_keys_to_load[0]]:
             raise ValueError("No trajectories loaded. Check your HDF5 file path and view keys.")
        self.traj_lens = [len(traj) for traj in self.data_dict[self.view_keys_to_load[0]]]

        min_traj_len = min(self.traj_lens) if self.traj_lens else 0
        if min_traj_len <= max_offset:
            print(f"Warning: max_offset ({max_offset}) is not smaller than the minimum trajectory length ({min_traj_len}). Adjusting max_offset.")
            self.max_offset = min_traj_len -1
        else:
            self.max_offset = max_offset
        
        # Set img_hw from resize_wh
        self.img_hw = self.resize_wh[0]  # Assuming square images
        
        # Calculate total valid moments (transitions) across all trajectories
        self.total_moments = 0
        for traj_len in self.traj_lens:
            self.total_moments += max(0, traj_len - self.max_offset)
        
        # Create mapping from moment index to (traj_idx, transition_idx)
        self.moment_to_traj_transition = []
        for traj_idx, traj_len in enumerate(self.traj_lens):
            valid_transitions = max(0, traj_len - self.max_offset)
            for transition_idx in range(valid_transitions):
                self.moment_to_traj_transition.append((traj_idx, transition_idx))

        self.fixed_offset = None

    def set_fixed_offset(self, offset):
        """Set a fixed offset for all subsequent __getitem__ calls."""
        self.fixed_offset = offset

    def load_multi_view_data(self, hdf5_path, num_trajectories_to_load=None):
        """Load multi-view image data from HDF5 file, preserving trajectory structure."""
        print(f"Loading multi-view data from HDF5 file: {hdf5_path}...")
        try:
            with h5py.File(hdf5_path, 'r') as f:
                demos = sorted(list(f['data'].keys()))
                # Filter demos by pre-selected names if provided
                if self.selected_demo_names is not None:
                    demos = [d for d in demos if d in self.selected_demo_names]
                if num_trajectories_to_load is not None:
                    demos = demos[:num_trajectories_to_load]

                # Persist the demo order used in this dataset instance
                self.demo_names = list(demos)

                for demo in tqdm(demos, desc="Loading Multi-view Demos"):
                    temp_traj_data = {view: [] for view in self.view_keys_to_load}
                    
                    # First, check if all required views exist for this demo
                    is_demo_valid = True
                    for view_key in self.view_keys_to_load:
                        full_key = f'data/{demo}/obs/{view_key}'
                        if full_key not in f:
                            print(f"Warning: View {view_key} not found in demo {demo}. Skipping this demo.")
                            is_demo_valid = False
                            break
                    if not is_demo_valid:
                        continue

                    # Load images for the valid demo
                    for view_key in self.view_keys_to_load:
                        images = f[f'data/{demo}/obs/{view_key}'][()]
                        resized_images = [np.array(Image.fromarray(img).resize(self.resize_wh)) for img in images]
                        temp_traj_data[view_key] = np.stack(resized_images)
                    
                    # Append trajectory data for all views
                    for view_key in self.view_keys_to_load:
                        self.data_dict[view_key].append(temp_traj_data[view_key])

        except Exception as e:
            print(f"Error loading HDF5 file: {e}")
            raise
        
        print("Multi-view data loaded successfully.")

    def load_actions_states(self, hdf5_path, num_trajectories_to_load=None):
        """Load actions and states from HDF5 file, preserving trajectory structure."""
        print("Loading actions and states...")
        try:
            with h5py.File(hdf5_path, 'r') as f:
                demos = sorted(list(f['data'].keys()))
                # Apply the same demo filtering
                if self.selected_demo_names is not None:
                    demos = [d for d in demos if d in self.selected_demo_names]
                if num_trajectories_to_load is not None:
                    demos = demos[:num_trajectories_to_load]

                # Keep the same order with images
                if hasattr(self, 'demo_names'):
                    demos = [d for d in self.demo_names if d in set(demos)]

                for demo in tqdm(demos, desc="Loading Actions and States"):
                    # A simple check to ensure we only load actions/states for demos that had all views
                    is_demo_valid = all(f'data/{demo}/obs/{view_key}' in f for view_key in self.view_keys_to_load)
                    if not is_demo_valid:
                        continue

                    # Load actions
                    if f'data/{demo}/actions' in f:
                        actions = f[f'data/{demo}/actions'][()]
                        self.actions.append(actions)
                    
                    # Load states  
                    if f'data/{demo}/states' in f:
                        states = f[f'data/{demo}/states'][()]
                        self.states.append(states)
                
                # Get dimensions from the first trajectory
                if self.actions:
                    self.act_dim = self.actions[0].shape[-1]
                if self.states:
                    self.state_dim = self.states[0].shape[-1]
                
                # Load normalization parameters if available
                if "action_mean" in f.attrs:
                    self.action_mean = f.attrs["action_mean"]
                    print("INFO: self.action_mean", self.action_mean)
                else:
                    self.action_mean = None
                if "action_std" in f.attrs:
                    self.action_std = f.attrs["action_std"]
                    print("INFO: self.action_std", self.action_std)
                else:
                    self.action_std = None
                    
        except Exception as e:
            print(f"Error loading actions/states: {e}")
            raise

    def _get_single_view_padded_obs(self, traj_idx, idx, view_key):
        """Get single view observation with frame stacking and padding."""
        # Get observations for this view and trajectory
        traj_view_obs = self.data_dict[view_key][traj_idx]
        traj_len = len(traj_view_obs)
        
        # Handle empty trajectory
        if traj_len == 0:
            # Create dummy observation with proper shape (H, W, T*C)
            dummy_shape = (*self.resize_wh, self.frame_stack * 3)
            return torch.zeros(dummy_shape, device=self.device, dtype=torch.uint8)
        
        # Limit index to trajectory length
        idx = min(idx, traj_len - 1)
        
        # Frame stacking (same as original DCSLAOMInMemoryDataset)
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = min(idx + 1, traj_len)
        obs = traj_view_obs[min_obs_idx:max_obs_idx]

        # Pad if at the beginning (same as original)
        if len(obs) < self.frame_stack:
            pad_img = obs[0][None]
            obs = np.concatenate([pad_img] * (self.frame_stack - len(obs)) + [obs])
        
        # Convert to tensor and reshape: (T, H, W, C) -> (H, W, T*C)
        obs = torch.as_tensor(np.array(obs), device=self.device)
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)
        
        return obs

    def __len__(self):
        return self.total_moments

    def __getitem__(self, idx):
        # idx는 이제 '전체 데이터셋에서 몇 번째 순간인가'를 의미합니다.
        if idx >= len(self.moment_to_traj_transition):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.moment_to_traj_transition)} moments")
        
        # 1. idx를 (traj_idx, transition_idx)로 변환
        traj_idx, transition_idx = self.moment_to_traj_transition[idx]
        
        # Calculate offset once per moment
        current_traj_len = self.traj_lens[traj_idx]
        max_possible_offset = current_traj_len - transition_idx - 1
        
        if self.fixed_offset is not None:
            # Ensure the fixed offset is valid for this transition
            offset = min(self.fixed_offset, max_possible_offset)
        else:
            offset = random.randint(1, min(self.max_offset, max_possible_offset))

        # Generate ALL possible view pairs (collate_fn will select K pairs for the batch)
        view_key_to_id = {key: idx for idx, key in enumerate(self.view_keys_to_load)}
        all_pairs = [(v1, v2) for v1 in self.view_keys_to_load for v2 in self.view_keys_to_load]
        
        view_ids_v1 = []
        view_ids_v2 = []
        obs_v1_list = []
        obs_v2_list = []
        future_obs_v1_list = []
        future_obs_v2_list = []
        
        for v1_key, v2_key in all_pairs:
            view_ids_v1.append(view_key_to_id[v1_key])
            view_ids_v2.append(view_key_to_id[v2_key])
            
            obs_v1 = self._get_single_view_padded_obs(traj_idx, transition_idx, v1_key)
            obs_v2 = self._get_single_view_padded_obs(traj_idx, transition_idx, v2_key)
            future_obs_v1 = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, v1_key)
            future_obs_v2 = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, v2_key)
            
            obs_v1_list.append(obs_v1)
            obs_v2_list.append(obs_v2)
            future_obs_v1_list.append(future_obs_v1)
            future_obs_v2_list.append(future_obs_v2)

        # 3. 공통 정보 (액션, 상태, ID) 가져오기
        action = torch.tensor(self.actions[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
        state = torch.tensor(self.states[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
        
        # 액션 시퀀스 추가 (offset 만큼의 액션들을 concatenate)
        action_sequence = self.actions[traj_idx][transition_idx : transition_idx + offset]
        action_sequence = torch.tensor(action_sequence, device=self.device, dtype=torch.float32)
        
        # 이 '순간'의 고유 ID
        instance_id = traj_idx * 10000 + transition_idx
        num_pairs = len(all_pairs)

        return {
            "obs_v1": torch.stack(obs_v1_list),  # [num_all_pairs, H, W, T*C]
            "obs_v2": torch.stack(obs_v2_list),
            "future_obs_v1": torch.stack(future_obs_v1_list),
            "future_obs_v2": torch.stack(future_obs_v2_list),
            
            # Pair 단위 metadata
            "view_ids_v1": torch.tensor(view_ids_v1, device=self.device),
            "view_ids_v2": torch.tensor(view_ids_v2, device=self.device),
            "instance_ids": torch.tensor([instance_id] * num_pairs, device=self.device),
            "offsets": torch.tensor([offset - 1] * num_pairs, device=self.device),
            
            # Scalar (collate에서 처리)
            "action": action,
            "action_sequence": action_sequence,
            "state": state,
        }


class DCSViewActionTrueActionsDataset(IterableDataset):
    _debug_counter = 0  # Class variable for debugging
    
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1, camera_num=8, view_keys=None, resize_wh=(128, 128), view_keys_to_load=None, num_trajectories_to_load=None, selected_demo_names=None):
        """
        Multi-view DCS dataset for view-action pair learning with infinite random sampling.
        Generates view pairs for contrastive learning.
        """
        self.frame_stack = frame_stack
        self.device = device
        self.max_offset = max_offset
        self.camera_num = camera_num
        self.view_keys = view_keys or []
        self.resize_wh = resize_wh
        
        self.view_keys_to_load = view_keys_to_load if view_keys_to_load is not None else self.view_keys
        # Pre-selected demo names to restrict random sampling space
        self.selected_demo_names = set(selected_demo_names) if selected_demo_names is not None else None
        
        self.data_dict = {view: [] for view in self.view_keys_to_load}
        self.load_multi_view_data(hdf5_path, num_trajectories_to_load)
        
        self.actions = []
        self.states = []
        self.load_actions_states(hdf5_path, num_trajectories_to_load)
        
        if not self.view_keys_to_load or not self.data_dict[self.view_keys_to_load[0]]:
            raise ValueError("No trajectories loaded for Labeled Dataset. Check HDF5 path and view keys.")
        
        self.traj_lens = [len(traj) for traj in self.data_dict[self.view_keys_to_load[0]]]
        
        min_traj_len = min(self.traj_lens) if self.traj_lens else 0
        if min_traj_len <= max_offset:
            print(f"Warning (Labeled Dataset): max_offset ({max_offset}) is not smaller than min_traj_len ({min_traj_len}). Adjusting.")
            self.max_offset = min_traj_len - 1
        else:
            self.max_offset = max_offset
            
        self.img_hw = self.resize_wh[0]
        
        self.fixed_offset = None

    def set_fixed_offset(self, offset):
        self.fixed_offset = offset

    def load_multi_view_data(self, hdf5_path, num_trajectories_to_load=None):
        DCSMVInMemoryDataset.load_multi_view_data(self, hdf5_path, num_trajectories_to_load)

    def load_actions_states(self, hdf5_path, num_trajectories_to_load=None):
        DCSMVInMemoryDataset.load_actions_states(self, hdf5_path, num_trajectories_to_load)

    def _get_single_view_padded_obs(self, traj_idx, idx, view_key):
        return DCSMVInMemoryDataset._get_single_view_padded_obs(self, traj_idx, idx, view_key)

    def __iter__(self):
        while True:
            # Sample a trajectory respecting selected_demo_names if provided
            if self.selected_demo_names is not None:
                # Build a mapping from demo name order to index once
                pass
            traj_idx = random.randint(0, len(self.traj_lens) - 1)
            traj_len = self.traj_lens[traj_idx]
            
            if traj_len <= self.max_offset:
                continue
            
            transition_idx = random.randint(0, traj_len - self.max_offset - 1)

            max_possible_offset = traj_len - transition_idx - 1
            if self.fixed_offset is not None:
                offset = min(self.fixed_offset, max_possible_offset)
            else:
                offset = random.randint(1, min(self.max_offset, max_possible_offset))

            # Generate ALL possible view pairs (collate_fn will select K pairs for the batch)
            view_key_to_id = {key: idx for idx, key in enumerate(self.view_keys_to_load)}
            all_pairs = [(v1, v2) for v1 in self.view_keys_to_load for v2 in self.view_keys_to_load]
            
            view_ids_v1 = []
            view_ids_v2 = []
            obs_v1_list = []
            obs_v2_list = []
            future_obs_v1_list = []
            future_obs_v2_list = []
            
            for v1_key, v2_key in all_pairs:
                view_ids_v1.append(view_key_to_id[v1_key])
                view_ids_v2.append(view_key_to_id[v2_key])
                
                obs_v1 = self._get_single_view_padded_obs(traj_idx, transition_idx, v1_key)
                obs_v2 = self._get_single_view_padded_obs(traj_idx, transition_idx, v2_key)
                future_obs_v1 = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, v1_key)
                future_obs_v2 = self._get_single_view_padded_obs(traj_idx, transition_idx + offset, v2_key)
                
                obs_v1_list.append(obs_v1)
                obs_v2_list.append(obs_v2)
                future_obs_v1_list.append(future_obs_v1)
                future_obs_v2_list.append(future_obs_v2)

            action = torch.tensor(self.actions[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
            state = torch.tensor(self.states[traj_idx][transition_idx], device=self.device, dtype=torch.float32)
            
            action_sequence_np = self.actions[traj_idx][transition_idx : transition_idx + offset]
            action_sequence = torch.as_tensor(action_sequence_np, device=self.device, dtype=torch.float32).reshape(-1)

            # 같은 trajectory+transition에서 생성된 모든 positive samples는 같은 instance_id를 가짐
            instance_id = traj_idx * 10000 + transition_idx
            num_pairs = len(all_pairs)

            DCSViewActionTrueActionsDataset._debug_counter += 1
            
            yield {
                "obs_v1": torch.stack(obs_v1_list),
                "obs_v2": torch.stack(obs_v2_list),
                "future_obs_v1": torch.stack(future_obs_v1_list),
                "future_obs_v2": torch.stack(future_obs_v2_list),
                
                # Pair 단위 metadata
                "view_ids_v1": torch.tensor(view_ids_v1, device=self.device),
                "view_ids_v2": torch.tensor(view_ids_v2, device=self.device),
                "instance_ids": torch.tensor([instance_id] * num_pairs, device=self.device),
                "offsets": torch.tensor([offset - 1] * num_pairs, device=self.device),
                
                # Scalar (collate에서 처리)
                "action": action,
                "action_sequence": action_sequence,
                "state": state,
            }


class DCSLAOMTrueActionsDataset(IterableDataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.states = [torch.tensor(df[traj]["states"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]
            self.state_dim = self.states[0][0].shape[-1]

            if "action_mean" in df.attrs:
                self.action_mean = df.attrs["action_mean"]
                print("INFO: self.action_mean", self.action_mean)
            else:
                self.action_mean = None
            if "action_std" in df.attrs:
                self.action_std = df.attrs["action_std"]
                print("INFO: self.action_std", self.action_std)
            else:
                self.action_std = None

            if self.action_mean is not None and self.action_std is not None:
                self.actions = (self.actions - self.action_mean) / self.action_std

        self.frame_stack = frame_stack
        # 각 trajectory의 길이를 개별적으로 저장
        self.traj_lens = [obs.shape[0] for obs in self.observations]

        # if "action_mean" in df.attrs:
        #     self.action_mean = df.attrs["action_mean"]
        #     print("INFO: self.action_mean", self.action_mean)
        # else:
        #     self.action_mean = None
        # if "action_std" in df.attrs: 
        #     self.action_std = df.attrs["action_std"]
        #     print("INFO: self.action_std", self.action_std)
        # else:
        #     self.action_std = None

        self.action_mean = None
        self.action_std = None

        self.new_actions = []
        if self.action_mean is not None and self.action_std is not None:
            for act in self.actions:
                act = (act - self.action_mean) / self.action_std
                self.new_actions.append(act)
            self.actions = self.new_actions
        
        # 최소 길이를 기준으로 max_offset 설정
        min_traj_len = min(self.traj_lens)
        assert 1 <= max_offset < min_traj_len
        self.max_offset = max_offset
        
        # 누적 길이를 미리 계산하여 저장
        self.cumulative_transitions = []
        total = 0
        for traj_len in self.traj_lens:
            valid_transitions = max(0, traj_len - self.max_offset)
            total += valid_transitions
            self.cumulative_transitions.append(total)

    def __get_padded_obs(self, traj_idx, idx):
        # 해당 trajectory의 실제 길이 사용
        traj_len = self.traj_lens[traj_idx]
        
        # 인덱스가 trajectory 길이를 초과하지 않도록 제한
        idx = min(idx, traj_len - 1)
        
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = min(idx + 1, traj_len)
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]
        
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)
        return obs

    def __iter__(self):
        while True:
            traj_idx = random.randint(0, len(self.actions) - 1)
            traj_len = self.traj_lens[traj_idx]
            
            # 해당 trajectory에서 유효한 transition 범위 계산
            max_valid_transition = traj_len - self.max_offset
            if max_valid_transition <= 0:
                continue  # 너무 짧은 trajectory는 건너뛰기
                
            transition_idx = random.randint(0, max_valid_transition - 1)

            obs = self.__get_padded_obs(traj_idx, transition_idx)
            next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
            offset = random.randint(1, min(self.max_offset, traj_len - transition_idx - 1))
            future_obs = self.__get_padded_obs(traj_idx, transition_idx + offset)

            action = self.actions[traj_idx][transition_idx]
            state = self.states[traj_idx][transition_idx]

            yield obs, next_obs, future_obs, action, state, (offset - 1)


class DCSDecoderInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, offset=1, device="cpu"):
        self.offset = offset
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1] * self.offset

            if "action_mean" in df.attrs:
                self.action_mean = df.attrs["action_mean"]
                print("INFO: self.action_mean", self.action_mean)
            else:
                self.action_mean = None
            if "action_std" in df.attrs:
                self.action_std = df.attrs["action_std"]
                print("INFO: self.action_std", self.action_std)
            else:
                self.action_std = None
            self.new_actions = []
            if self.action_mean is not None and self.action_std is not None:
                for act in self.actions:
                    act = (act - self.action_mean) / self.action_std
                    self.new_actions.append(act)
                self.actions = self.new_actions

        self.frame_stack = frame_stack
        # 각 trajectory의 길이를 개별적으로 저장
        self.traj_lens = [obs.shape[0] for obs in self.observations]
        
        # 누적 길이를 미리 계산하여 저장
        self.cumulative_transitions = []
        total = 0
        for traj_len in self.traj_lens:
            valid_transitions = max(0, traj_len - self.offset)
            total += valid_transitions
            self.cumulative_transitions.append(total)

    def __get_padded_obs(self, traj_idx, idx):
        # 해당 trajectory의 실제 길이 사용
        traj_len = self.traj_lens[traj_idx]
        
        # 인덱스가 trajectory 길이를 초과하지 않도록 제한
        idx = min(idx, traj_len - 1)
        
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = min(idx + 1, traj_len)
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs

    def __len__(self):
        return self.cumulative_transitions[-1] if self.cumulative_transitions else 0

    def __getitem__(self, idx):
        # 누적 길이 배열에서 trajectory 찾기
        traj_idx = 0
        for i, cumulative in enumerate(self.cumulative_transitions):
            if idx < cumulative:
                traj_idx = i
                break
        
        # 해당 trajectory 내에서의 transition 인덱스 계산
        if traj_idx == 0:
            transition_idx = idx
        else:
            transition_idx = idx - self.cumulative_transitions[traj_idx - 1]

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + self.offset)
        action = self.actions[traj_idx][transition_idx : transition_idx + self.offset]
        action = action.reshape(-1)

        return obs, next_obs, action

def normalize_img(img):
    return ((img / 255.0) - 0.5) * 2.0


def unnormalize_img(img):
    return ((img / 2.0) + 0.5) * 255.0

def view_action_collate_fn(batch_list, K=4):
    """
    Collate function for view-action pair learning.
    Sample K pairs ONCE for the batch, then apply to all P instances → K*P batch.
    All instances in the batch use the SAME K view pairs.
    
    Args:
        batch_list: 배치 데이터 리스트, 각 아이템은 모든 가능한 view pair를 포함
        K: 배치 전체에서 사용할 pair 개수 (모든 instance가 공유)
    """
    P = len(batch_list)
    if P == 0:
        return None
    
    # 배치 단위로 K개의 view pair를 한 번만 샘플링
    num_available_pairs = batch_list[0]["obs_v1"].shape[0]
    k_sample = min(K, num_available_pairs)
    if k_sample == 0:
        return None
    
    # 배치 전체에서 사용할 K개의 pair indices를 한 번만 선택
    sampled_indices = random.sample(range(num_available_pairs), k_sample)
    
    final_obs_v1 = []
    final_obs_v2 = []
    final_future_obs_v1 = []
    final_future_obs_v2 = []
    final_view_ids_v1 = []
    final_view_ids_v2 = []
    final_instance_ids = []
    final_offsets = []
    final_actions = []
    unpadded_action_sequences = []
    final_states = []

    # 모든 instance에 같은 sampled_indices 적용
    for i in range(P):
        instance_group = batch_list[i]
        
        final_obs_v1.append(instance_group["obs_v1"][sampled_indices])
        final_obs_v2.append(instance_group["obs_v2"][sampled_indices])
        final_future_obs_v1.append(instance_group["future_obs_v1"][sampled_indices])
        final_future_obs_v2.append(instance_group["future_obs_v2"][sampled_indices])
        final_view_ids_v1.append(instance_group["view_ids_v1"][sampled_indices])
        final_view_ids_v2.append(instance_group["view_ids_v2"][sampled_indices])
        final_instance_ids.append(instance_group["instance_ids"][sampled_indices])
        final_offsets.append(instance_group["offsets"][sampled_indices])
        
        # action, state는 scalar를 k_sample만큼 복제
        final_actions.append(instance_group["action"].unsqueeze(0).repeat(k_sample, 1))
        final_states.append(instance_group["state"].unsqueeze(0).repeat(k_sample, 1))
        
        # action_sequence도 샘플링된 pair만큼 추가
        for _ in range(k_sample):
            unpadded_action_sequences.append(instance_group["action_sequence"])

    if not final_obs_v1:
        # Handle case where batch is empty after filtering
        return None

    # action_sequences 패딩 처리
    padded_action_sequences = pad_sequence(unpadded_action_sequences, batch_first=True, padding_value=0.0)
    
    # 패딩 마스크 생성 (1: 실제 데이터, 0: 패딩)
    action_sequences_mask = torch.zeros(padded_action_sequences.shape[:2], device=padded_action_sequences.device)
    for i, seq in enumerate(unpadded_action_sequences):
        action_sequences_mask[i, :len(seq)] = 1.0

    # (B, L, D) -> (B, L*D) 형태로 flatten
    flat_padded_action_sequences = padded_action_sequences.flatten(start_dim=1)
    
    # view_pair_ids 생성 (v1 * 10000 + v2)
    view_ids_v1_cat = torch.cat(final_view_ids_v1, dim=0)
    view_ids_v2_cat = torch.cat(final_view_ids_v2, dim=0)
    view_pair_ids = view_ids_v1_cat * 10000 + view_ids_v2_cat

    return {
        "obs_v1": torch.cat(final_obs_v1, dim=0),  # [K*P, H, W, C]
        "obs_v2": torch.cat(final_obs_v2, dim=0),
        "future_obs_v1": torch.cat(final_future_obs_v1, dim=0),
        "future_obs_v2": torch.cat(final_future_obs_v2, dim=0),
        
        "view_pair_ids": view_pair_ids,  # [K*P]
        "instance_ids": torch.cat(final_instance_ids, dim=0),
        "offsets": torch.cat(final_offsets, dim=0),
        
        "actions": torch.cat(final_actions, dim=0),
        "action_sequences": flat_padded_action_sequences,
        "action_sequences_mask": action_sequences_mask,
        "states": torch.cat(final_states, dim=0),
    }


def metric_learning_collate_fn(batch_list, K=4, mixed_view_sampling=False):
    """
    Dataloader로부터 받은 인스턴스 그룹 리스트(batch_list)를 받아,
    P개의 인스턴스에서 각각 K개의 뷰를 샘플링하여 최종 배치를 구성합니다.
    action_sequences에 대해 패딩을 적용하여 가변 길이를 처리합니다.
    
    Args:
        batch_list: 배치 데이터 리스트
        K: 인스턴스당 샘플링할 뷰 개수 (mixed_view_sampling=True일 때는 무시됨)
        mixed_view_sampling: True면 각 인스턴스에서 단일 observation 사용
    """
    P = len(batch_list)
    
    final_obs = []
    final_next_obs = []
    final_future_obs = []
    final_actions = []
    unpadded_action_sequences = []
    final_states = []
    final_offsets = []
    final_instance_ids = []
    final_obs_camera_indices = []  # 새로 추가
    final_future_camera_indices = []  # 새로 추가

    for i in range(P):
        instance_group = batch_list[i]
        
        if mixed_view_sampling:
            # Mixed view sampling: 모든 positive samples 사용 (여러 mixed view 조합으로 생성됨)
            num_positive_samples = instance_group["obs"].shape[0]
            
            final_obs.append(instance_group["obs"])  # 모든 positive samples
            final_next_obs.append(instance_group["next_obs"])
            final_future_obs.append(instance_group["future_obs"])
            
            # Camera indices 추가
            final_obs_camera_indices.append(instance_group["obs_camera_idx"])
            final_future_camera_indices.append(instance_group["future_camera_idx"])
            
            # 각 positive sample에 대해 동일한 action, state, offset, instance_id 할당
            final_actions.append(instance_group["action"].unsqueeze(0).repeat(num_positive_samples, 1))
            
            # 각 positive sample에 대해 동일한 action_sequence 추가
            for _ in range(num_positive_samples):
                unpadded_action_sequences.append(instance_group["action_sequence"])
            
            final_states.append(instance_group["state"].unsqueeze(0).repeat(num_positive_samples, 1))
            final_offsets.append(torch.tensor([instance_group["offset"]] * num_positive_samples, device=instance_group["action"].device))
            final_instance_ids.append(torch.tensor([instance_group["instance_id"]] * num_positive_samples))
        else:
            # Original behavior: K개 뷰 샘플링
            num_available_views = instance_group["obs"].shape[0]
            # k_sample = min(K, num_available_views)
            k_sample = K
            # if K > num_available_views:
                # print(f"INFO: overriding K: {K} to num_available_views: {num_available_views}")
            if k_sample == 0:
                continue

            if K > num_available_views:
                sampled_indices = random.choices(range(num_available_views), k=k_sample)
            else:
                sampled_indices = random.sample(range(num_available_views), k_sample)
            
            final_obs.append(instance_group["obs"][sampled_indices])
            final_next_obs.append(instance_group["next_obs"][sampled_indices])
            final_future_obs.append(instance_group["future_obs"][sampled_indices])
            
            # Camera indices 추가 (샘플링된 인덱스 사용)
            final_obs_camera_indices.append(instance_group["obs_camera_idx"][sampled_indices])
            final_future_camera_indices.append(instance_group["future_camera_idx"][sampled_indices])
            
            final_actions.append(instance_group["action"].unsqueeze(0).repeat(k_sample, 1))
            
            # 각 뷰에 대해 동일한 action_sequence를 추가
            for _ in range(k_sample):
                unpadded_action_sequences.append(instance_group["action_sequence"])

            final_states.append(instance_group["state"].unsqueeze(0).repeat(k_sample, 1))
            final_offsets.append(torch.tensor([instance_group["offset"]] * k_sample, device=instance_group["action"].device))
            final_instance_ids.append(torch.tensor([instance_group["instance_id"]] * k_sample))

    if not final_obs:
        # Handle case where batch is empty after filtering
        return None

    # action_sequences 패딩 처리
    padded_action_sequences = pad_sequence(unpadded_action_sequences, batch_first=True, padding_value=0.0)
    
    # 패딩 마스크 생성 (1: 실제 데이터, 0: 패딩)
    action_sequences_mask = torch.zeros(padded_action_sequences.shape[:2], device=padded_action_sequences.device)
    for i, seq in enumerate(unpadded_action_sequences):
        action_sequences_mask[i, :len(seq)] = 1.0

    # (B, L, D) -> (B, L*D) 형태로 flatten
    flat_padded_action_sequences = padded_action_sequences.flatten(start_dim=1)

    return {
        "obs": torch.cat(final_obs, dim=0),
        "next_obs": torch.cat(final_next_obs, dim=0),
        "future_obs": torch.cat(final_future_obs, dim=0),
        "actions": torch.cat(final_actions, dim=0),
        "action_sequences": flat_padded_action_sequences,
        "action_sequences_mask": action_sequences_mask,
        "states": torch.cat(final_states, dim=0),
        "offsets": torch.cat(final_offsets, dim=0),
        "instance_ids": torch.cat(final_instance_ids, dim=0),
        "obs_camera_ids": torch.cat(final_obs_camera_indices, dim=0),  # 새로 추가
        "future_camera_ids": torch.cat(final_future_camera_indices, dim=0),  # 새로 추가
    }


def evaluation_collate_fn(batch_list):
    """
    Collate function for evaluation. Does not perform random sampling.
    It selects the first available view from each instance group.
    """
    if not batch_list:
        return None

    # Use the same keys as the training collate function
    keys = ["obs", "next_obs", "future_obs", "actions", "action_sequences", "states", "offsets", "instance_ids"]
    final_batch = {key: [] for key in keys}

    for instance_group in batch_list:
        # Select the first available view (index 0)
        final_batch["obs"].append(instance_group["obs"][0].unsqueeze(0))
        final_batch["next_obs"].append(instance_group["next_obs"][0].unsqueeze(0))
        final_batch["future_obs"].append(instance_group["future_obs"][0].unsqueeze(0))
        
        # Other tensors are the same for all views
        final_batch["actions"].append(instance_group["action"].unsqueeze(0))
        final_batch["action_sequences"].append(instance_group["action_sequence"].unsqueeze(0))
        final_batch["states"].append(instance_group["state"].unsqueeze(0))
        final_batch["offsets"].append(torch.tensor([instance_group["offset"]]))
        final_batch["instance_ids"].append(torch.tensor([instance_group["instance_id"]]))

    # Concatenate all items into a single batch tensor
    try:
        return {key: torch.cat(final_batch[key], dim=0) for key in keys}
    except RuntimeError:
        # This can happen if the batch is empty or dimensions mismatch
        return None


def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class SelectPixelsObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["pixels"]

    def observation(self, obs):
        return obs["pixels"]


class FlattenStackedFrames(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_shape = self.env.observation_space.shape
        new_shape = old_shape[1:-1] + (old_shape[0] * old_shape[-1],)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        obs = obs.transpose((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)
        return obs


def create_env_from_df(
    hdf5_path,
    backgrounds_path,
    backgrounds_split,
    frame_stack=1,
    pixels_only=True,
    flatten_frames=True,
    difficulty=None,
):
    with h5py.File(hdf5_path, "r") as df:
        dm_env = suite.load(
            domain_name=df.attrs["domain_name"],
            task_name=df.attrs["task_name"],
            difficulty=df.attrs["difficulty"] if difficulty is None else difficulty,
            dynamic=df.attrs["dynamic"],
            background_dataset_path=backgrounds_path,
            background_dataset_videos=backgrounds_split,
            pixels_only=pixels_only,
            render_kwargs=dict(height=df.attrs["img_hw"], width=df.attrs["img_hw"]),
        )
        env = DmControlCompatibilityV0(dm_env)
        env = gym.wrappers.ClipAction(env)

        if pixels_only:
            env = SelectPixelsObsWrapper(env)

        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
            if flatten_frames:
                env = FlattenStackedFrames(env)

    return env


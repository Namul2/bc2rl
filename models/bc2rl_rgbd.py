import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from behavior_cloning.evaluate import evaluate
from behavior_cloning.make_env import make_eval_envs


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "BC2RL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""

    # Environment
    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode to use for the environments"""
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps"""
    sim_backend: str = "gpu"
    """the simulation backend for environments"""
    include_state: bool = True
    """whether to include state information with RGB observations"""
    use_depth: bool = True
    """whether to use depth channel along with RGB"""

    # BC pretraining arguments
    demo_path: str = "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.rgbd.pd_ee_delta_pose.h5"
    """the path of demo dataset"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    bc_epochs: int = 50
    """number of epochs for BC pretraining"""
    bc_batch_size: int = 64
    """batch size for BC pretraining"""
    bc_lr: float = 1e-3
    """learning rate for BC pretraining"""
    
    # RL training arguments
    total_timesteps: int = 10_000_000
    """total timesteps for RL training"""
    learning_rate: float = 3e-4
    """learning rate for RL training"""
    num_envs: int = 256
    """number of parallel environments"""
    num_steps: int = 50
    """number of steps per rollout"""
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """lambda for GAE"""
    num_minibatches: int = 32
    """number of mini-batches"""
    update_epochs: int = 4
    """number of epochs per PPO update"""
    norm_adv: bool = True
    """normalize advantages"""
    clip_coef: float = 0.2
    """PPO clipping coefficient"""
    clip_vloss: bool = False
    """clip value loss"""
    ent_coef: float = 0.01
    """entropy coefficient"""
    vf_coef: float = 0.5
    """value function coefficient"""
    max_grad_norm: float = 0.5
    """max gradient norm"""
    target_kl: Optional[float] = None
    """target KL divergence"""
    
    # BC2RL specific arguments
    bc_reg_coef: float = 0.5
    """BC regularization coefficient for RL training"""
    bc_reg_decay: bool = True
    """whether to decay BC regularization coefficient"""
    bc_reg_min: float = 0.0
    """minimum BC regularization coefficient"""
    bc_reg_decay_steps: int = 5_000_000
    """steps over which to decay BC regularization"""
    
    # Evaluation arguments
    num_eval_envs: int = 10
    """number of parallel evaluation environments"""
    num_eval_episodes: int = 100
    """number of episodes for evaluation"""
    eval_freq: int = 25
    """evaluation frequency (in iterations)"""
    save_freq: int = 50
    """model save frequency (in iterations)"""
    
    # Computed in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class PlainConv(nn.Module):
    """CNN encoder for RGB/RGBD images"""
    def __init__(
        self,
        in_channels=4,
        out_dim=256,
        max_pooling=True,
        inactivated_output=False,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 64]
            nn.Conv2d(16, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=not inactivated_output)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class DictArray:
    """Helper class for handling dictionary observations"""
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = torch.float32 if v.dtype in (np.float32, np.float64) else torch.uint8
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        else:
            for k, v in value.items():
                self.data[k][index] = v

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class BCAgent(nn.Module):
    """Agent for BC pretraining with visual inputs"""
    def __init__(self, state_dim: int, action_dim: int, num_cameras: int = 1, use_depth: bool = True):
        super().__init__()
        in_channels = (4 if use_depth else 3) * num_cameras
        self.encoder = PlainConv(
            in_channels=in_channels, 
            out_dim=256, 
            max_pooling=False, 
            inactivated_output=False
        )
        self.final_mlp = make_mlp(256 + state_dim, [512, 256, action_dim], last_act=False)
        
    def forward(self, rgbd, state):
        img = rgbd.permute(0, 3, 1, 2)  # (B, C, H, W)
        feature = self.encoder(img)
        x = torch.cat([feature, state], dim=1)
        return self.final_mlp(x)


class PPOAgent(nn.Module):
    """Agent for PPO training with visual inputs and BC regularization"""
    def __init__(self, envs, sample_obs, bc_agent: Optional[BCAgent] = None, use_depth: bool = True):
        super().__init__()
        
        # Calculate number of cameras
        if "rgb" in sample_obs:
            rgb_channels = sample_obs["rgb"].shape[-1]
            num_cameras = rgb_channels // 3
        else:
            num_cameras = 1
        
        in_channels = (4 if use_depth else 3) * num_cameras
        state_dim = sample_obs["state"].shape[-1] if "state" in sample_obs else 0
        act_dim = np.prod(envs.single_action_space.shape)
        
        # Initialize encoder and actor from BC agent if provided
        if bc_agent is not None:
            self.encoder = bc_agent.encoder
            self.actor_mean = bc_agent.final_mlp
        else:
            self.encoder = PlainConv(
                in_channels=in_channels,
                out_dim=256,
                max_pooling=False,
                inactivated_output=False
            )
            self.actor_mean = make_mlp(256 + state_dim, [512, 256, act_dim], last_act=False)
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(256 + state_dim, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        
        # Store BC agent for regularization
        self.bc_agent = bc_agent
        if self.bc_agent:
            for param in self.bc_agent.parameters():
                param.requires_grad = False
    
    def get_features(self, obs):
        rgbd = obs.get("rgbd") if "rgbd" in obs else obs.get("rgb")
        state = obs.get("state", None)
        
        img = rgbd.permute(0, 3, 1, 2) / 255.0
        feature = self.encoder(img)
        
        if state is not None:
            return torch.cat([feature, state], dim=1)
        return feature
    
    def get_value(self, obs):
        features = self.get_features(obs)
        return self.critic(features)
    
    def get_action(self, obs, deterministic=False):
        features = self.get_features(obs)
        state_dim = obs["state"].shape[-1] if "state" in obs else 0
        
        if state_dim > 0:
            action_mean = self.actor_mean(features)
        else:
            action_mean = self.actor_mean(features)
        
        if deterministic:
            return action_mean
        
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    
    def get_action_and_value(self, obs, action=None):
        features = self.get_features(obs)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)
    
    def get_bc_loss(self, obs, actions):
        """Compute BC regularization loss"""
        if self.bc_agent is None:
            return torch.tensor(0.0)
        
        rgbd = obs.get("rgbd") if "rgbd" in obs else obs.get("rgb")
        state = obs.get("state", torch.zeros((rgbd.shape[0], 0), device=rgbd.device))
        
        with torch.no_grad():
            bc_actions = self.bc_agent(rgbd, state)
        
        features = self.get_features(obs)
        pred_actions = self.actor_mean(features)
        
        return F.mse_loss(pred_actions, bc_actions)


class ManiSkillRGBDDataset(Dataset):
    """Dataset for loading RGBD demonstrations"""
    def __init__(self, dataset_file: str, device, load_count=None, use_depth=True):
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        
        self.camera_data = defaultdict(list)
        self.actions = []
        self.states = []
        self.device = device
        self.use_depth = use_depth
        
        if load_count is None:
            load_count = len(self.episodes)
        
        print(f"Loading {load_count} episodes for BC pretraining...")
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            
            # Handle both direct access and nested group structure
            if "obs" in trajectory:
                obs = trajectory["obs"]
            else:
                obs = trajectory
            
            # Extract state information
            if "agent" in obs and "extra" in obs:
                agent = obs["agent"]
                extra = obs["extra"]
                state = self._flatten_state_dict(agent, extra)
                self.states.append(state[:-1])  # Remove last obs
            
            # Extract camera data
            if "sensor_data" in obs:
                for cam_name, cam_data in obs["sensor_data"].items():
                    self.camera_data[cam_name + "_rgb"].append(cam_data["rgb"][:-1])
                    if self.use_depth and "depth" in cam_data:
                        self.camera_data[cam_name + "_depth"].append(cam_data["depth"][:-1])
            
            # Extract actions
            if "actions" in trajectory:
                self.actions.append(trajectory["actions"][:])
        
        # Stack all data
        for key in self.camera_data.keys():
            if "rgb" in key:
                self.camera_data[key] = np.vstack(self.camera_data[key]) / 255.0
            elif "depth" in key:
                self.camera_data[key] = np.vstack(self.camera_data[key]) / 1024.0
        
        if self.states:
            self.states = np.vstack(self.states)
        else:
            # If no state, create dummy states
            self.states = np.zeros((len(self.actions[0]), 0))
        
        self.actions = np.vstack(self.actions)
    
    def _flatten_state_dict(self, agent, extra):
        """Flatten state dictionaries into a single array"""
        states = []
        
        def flatten_dict(d):
            flat = []
            for k, v in d.items():
                if isinstance(v, (dict, h5py.Group)):
                    flat.extend(flatten_dict(v))
                else:
                    data = v[:] if hasattr(v, 'shape') else v
                    if isinstance(data, np.ndarray):
                        if data.ndim == 1:
                            flat.append(data)
                        else:
                            flat.append(data.reshape(data.shape[0], -1))
            return flat
        
        agent_flat = flatten_dict(agent)
        extra_flat = flatten_dict(extra)
        
        # Combine all state components
        all_states = agent_flat + extra_flat
        if all_states:
            return np.hstack(all_states)
        return np.zeros((len(self.actions), 0))
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        out = {}
        out["action"] = torch.from_numpy(self.actions[idx]).float().to(self.device)
        out["state"] = torch.from_numpy(self.states[idx]).float().to(self.device)
        
        rgbd_data = []
        for key in sorted(self.camera_data.keys()):
            if (not self.use_depth and "depth" in key):
                continue
            rgbd_data.append(torch.from_numpy(self.camera_data[key][idx]).float().to(self.device))
        out["rgbd"] = torch.cat(rgbd_data, dim=-1)
        
        return out


# def pretrain_bc(args, bc_agent, dataset, device):
#     """Pretrain agent with behavior cloning"""
#     print("\n" + "="*50)
#     print("Starting BC Pretraining with Visual Inputs...")
#     print("="*50)
    
#     optimizer = optim.Adam(bc_agent.parameters(), lr=args.bc_lr)
    
#     sampler = RandomSampler(dataset)
#     batch_sampler = BatchSampler(sampler, args.bc_batch_size, drop_last=True)
#     dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
    
#     bc_agent.train()
#     for epoch in range(args.bc_epochs):
#         epoch_losses = []
#         for batch in dataloader:
#             pred_actions = bc_agent(batch["rgbd"], batch["state"])
#             loss = F.mse_loss(pred_actions, batch["action"])
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             epoch_losses.append(loss.item())
        
#         avg_loss = np.mean(epoch_losses)
#         print(f"BC Epoch {epoch+1}/{args.bc_epochs}, Loss: {avg_loss:.4f}")
    
#     print("BC Pretraining completed!\n")
#     return bc_agent


def pretrain_bc(args, bc_agent, dataset, eval_envs, device, writer):
    """Pretrain agent with behavior cloning"""
    print("\n" + "="*50)
    print("Starting BC Pretraining with Visual Inputs...")
    print("="*50)
    
    optimizer = optim.Adam(bc_agent.parameters(), lr=args.bc_lr)
    
    sampler = RandomSampler(dataset)
    batch_sampler = BatchSampler(sampler, args.bc_batch_size, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
    
    best_bc_eval_success = 0.0
    global_bc_step = 0
    
    bc_agent.train()
    for epoch in range(args.bc_epochs):
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            pred_actions = bc_agent(batch["rgbd"], batch["state"])
            loss = F.mse_loss(pred_actions, batch["action"])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            global_bc_step += 1
            
            # Log training loss periodically
            if batch_idx % 10 == 0:
                writer.add_scalar("bc_pretrain/loss", loss.item(), global_bc_step)
        
        avg_loss = np.mean(epoch_losses)
        print(f"BC Epoch {epoch+1}/{args.bc_epochs}, Loss: {avg_loss:.4f}")
        writer.add_scalar("bc_pretrain/epoch_loss", avg_loss, epoch)
        
        # Evaluate BC agent periodically
        if (epoch + 1) % 25 == 0 and writer is not None:
            print(f"Evaluating BC agent at epoch {epoch+1}...")
            bc_agent.eval()
            
            def sample_fn(obs):
                # Handle both numpy and dict observations
                if isinstance(obs, dict):
                    for k, v in obs.items():
                        if isinstance(v, np.ndarray):
                            obs[k] = torch.from_numpy(v).float().to(device)
                    
                    # Extract rgbd and state
                    rgbd = obs.get("rgbd") if "rgbd" in obs else obs.get("rgb")
                    state = obs.get("state", torch.zeros((rgbd.shape[0], 0), device=device))
                    return bc_agent(rgbd, state)
                else:
                    # Fallback for unexpected format
                    return bc_agent(obs, torch.zeros((obs.shape[0], 0), device=device))
            
            with torch.no_grad():
                eval_metrics = evaluate(min(args.num_eval_episodes, 50), sample_fn, eval_envs)
            
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"bc_pretrain/eval_{k}", eval_metrics[k], epoch)
            
            success_rate = eval_metrics.get("success_at_end", 0.0)
            print(f"BC Eval success rate: {success_rate:.4f}")
            
            # Save best BC model
            if success_rate > best_bc_eval_success:
                best_bc_eval_success = success_rate
                save_path = f"runs/{run_name}/bc_best_eval.pt"
                torch.save(bc_agent.state_dict(), save_path)
                print(f"New best BC model saved (success: {success_rate:.4f})")
            
            bc_agent.train()
    
    print(f"BC Pretraining completed! Best eval success: {best_bc_eval_success:.4f}\n")
    return bc_agent


def train_rl(args, agent, envs, eval_envs, device, writer):
    """Train agent with PPO + BC regularization"""
    print("\n" + "="*50)
    print("Starting RL Training with BC Regularization...")
    print("="*50)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Start training
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    
    best_eval_success = 0.0
    
    for iteration in range(1, args.num_iterations + 1):
        # Compute current BC regularization coefficient
        if args.bc_reg_decay:
            decay_progress = min(global_step / args.bc_reg_decay_steps, 1.0)
            current_bc_coef = args.bc_reg_coef * (1 - decay_progress) + args.bc_reg_min * decay_progress
        else:
            current_bc_coef = args.bc_reg_coef
        
        # Rollout
        agent.eval()
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
            
            # Clip actions
            action_low = torch.from_numpy(envs.single_action_space.low).to(device)
            action_high = torch.from_numpy(envs.single_action_space.high).to(device)
            clipped_action = torch.clamp(action, action_low, action_high)
            
            next_obs, reward, terminations, truncations, infos = envs.step(clipped_action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)
            
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    writer.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                
                # Handle final observations
                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = \
                        agent.get_value(infos["final_observation"]).view(-1)
        
        # Compute advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values
        
        # Flatten batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # PPO update with BC regularization
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
                
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # BC regularization loss
                bc_loss = agent.get_bc_loss(b_obs[mb_inds], b_actions[mb_inds])
                
                # Total loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + current_bc_coef * bc_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # Logging
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/bc_loss", bc_loss.item(), global_step)
        writer.add_scalar("losses/bc_coef", current_bc_coef, global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        
        SPS = int(global_step / (time.time() - start_time))
        print(f"Iteration {iteration}/{args.num_iterations}, SPS: {SPS}, BC coef: {current_bc_coef:.4f}")
        
        # Evaluation
        if iteration % args.eval_freq == 0:
            print("Evaluating...")
            agent.eval()
            
            def sample_fn(obs):
                # Handle both numpy and tensor inputs
                if isinstance(obs, dict):
                    for k, v in obs.items():
                        if isinstance(v, np.ndarray):
                            obs[k] = torch.from_numpy(v).float().to(device)
                return agent.get_action(obs, deterministic=True)
            
            with torch.no_grad():
                eval_metrics = evaluate(args.num_eval_episodes, sample_fn, eval_envs)
            
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], global_step)
            
            success_rate = eval_metrics.get("success_at_end", 0.0)
            print(f"Eval success rate: {success_rate:.4f}")
            
            if success_rate > best_eval_success:
                best_eval_success = success_rate
                save_path = f"runs/{run_name}/best_model.pt"
                torch.save(agent.state_dict(), save_path)
                print(f"New best model saved: {save_path}")
        
        # Save checkpoint
        if iteration % args.save_freq == 0:
            save_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")
    
    # Save final model
    save_path = f"runs/{run_name}/final_model.pt"
    torch.save(agent.state_dict(), save_path)
    print(f"Final model saved: {save_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Compute runtime parameters
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Environment setup
    env_kwargs = dict(
        control_mode=args.control_mode,
        obs_mode="rgbd" if args.use_depth else "rgb",
        render_mode="all",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    
    # Training environments
    train_env_kwargs = env_kwargs.copy()
    train_env_kwargs["sim_backend"] = "gpu" if args.sim_backend == "gpu" else "cpu"
    
    envs = gym.make(args.env_id, num_envs=args.num_envs, **train_env_kwargs)
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=args.use_depth, state=args.include_state)
    
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    
    if args.capture_video:
        envs = RecordEpisode(
            envs,
            output_dir=f"runs/{run_name}/train_videos",
            save_trajectory=False,
            video_fps=30,
            save_video_trigger=lambda x: x % (args.eval_freq * args.num_steps) == 0,
            max_steps_per_video=args.num_steps
        )
    
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, record_metrics=True)
    
    # Evaluation environments
    eval_env_kwargs = env_kwargs.copy()
    # eval_env_kwargs["sim_backend"] = args.sim_backend
    
    eval_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        eval_env_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[lambda env: FlattenRGBDObservationWrapper(env, rgb=True, depth=args.use_depth, state=args.include_state)]
    )
    
    # Setup logging
    if args.track:
        import wandb
        config = vars(args)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="BC2RL-RGB",
            tags=["bc2rl", "visual", "bc_pretrain", "ppo_finetune"],
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Load dataset for BC pretraining
    dataset = ManiSkillRGBDDataset(
        args.demo_path, 
        device=device, 
        load_count=args.num_demos,
        use_depth=args.use_depth
    )
    
    # Calculate dimensions
    num_cameras = len([k for k in dataset.camera_data.keys() if "rgb" in k])
    state_dim = dataset.states.shape[1] if len(dataset.states.shape) > 1 else 0
    action_dim = envs.single_action_space.shape[0]
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Number of cameras: {num_cameras}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize BC agent and pretrain
    bc_agent = BCAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_cameras=num_cameras,
        use_depth=args.use_depth
    ).to(device)
    
    bc_agent = pretrain_bc(args, bc_agent, dataset, eval_envs, device, writer)
    
    # Save BC pretrained model
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    bc_save_path = f"runs/{run_name}/bc_pretrained.pt"
    torch.save(bc_agent.state_dict(), bc_save_path)
    print(f"BC pretrained model saved: {bc_save_path}")
    
    # Get sample observation for PPO agent initialization
    sample_obs, _ = envs.reset()
    
    # Initialize PPO agent with BC agent
    ppo_agent = PPOAgent(envs, sample_obs, bc_agent, use_depth=args.use_depth).to(device)
    
    # Train with RL + BC regularization
    train_rl(args, ppo_agent, envs, eval_envs, device, writer)
    
    # Cleanup
    envs.close()
    eval_envs.close()
    writer.close()
    
    if args.track:
        wandb.finish()
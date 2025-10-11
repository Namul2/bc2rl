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
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from behavior_cloning.evaluate import evaluate as bc_evaluate
from behavior_cloning.make_env import make_eval_envs

@dataclass
class Args:
    exp_name: Optional[str] = "bc2rl"
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "BC2RL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances"""
    
    # Environment
    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    control_mode: str = "pd_ee_delta_pose"
    """the control mode to use"""
    num_envs: int = 256
    """the number of parallel training environments"""
    num_eval_envs: int = 10
    """the number of parallel evaluation environments"""
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""

    
    # Demo data
    demo_path: str = "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5"
    """the path of demo dataset"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    
    # BC Pretraining
    bc_iterations: int = 50000
    """number of BC pretraining iterations"""
    bc_batch_size: int = 1024
    """batch size for BC training"""
    bc_lr: float = 3e-4
    """learning rate for BC"""
    bc_eval_freq: int = 5000
    """evaluation frequency during BC training"""
    sim_backend: str = "gpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu" """

    # RL Training
    rl_iterations: int = 5000
    """number of RL training iterations"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    
    # BC2RL specific
    demo_buffer_ratio: float = 0.2
    """ratio of demo data in each training batch (0.0 to 1.0)"""
    bc_reg_coef: float = 0.1
    """coefficient for BC regularization loss"""
    bc_reg_decay: float = 0.9995
    """decay rate for BC regularization coefficient"""
    use_bc_reg: bool = True
    """whether to use BC regularization"""
    use_demo_buffer: bool = True
    """whether to use demo buffer in RL training"""
    
    # Evaluation
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    num_eval_episodes: int = 100
    """number of episodes to evaluate"""
    
    # Other
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment"""
    save_freq: Optional[int] = None
    """checkpoint save frequency"""
    
    # Runtime filled
    batch_size: int = 0
    minibatch_size: int = 0


def load_h5_data(data):
    """Recursively load h5py data into dict"""
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class DemoBuffer:
    """Buffer to store and sample demonstration data"""
    def __init__(self, dataset_file: str, device, load_count=None):
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        
        if load_count is None:
            load_count = len(self.episodes)
        
        print(f"Loading {load_count} demo episodes...")
        observations = []
        actions = []
        
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            
            # Ignore last observation (terminal state)
            observations.append(trajectory["obs"][:-1])
            actions.append(trajectory["actions"])
        
        self.observations = torch.from_numpy(np.vstack(observations)).float().to(device)
        self.actions = torch.from_numpy(np.vstack(actions)).float().to(device)
        self.size = len(self.observations)
        
        print(f"Loaded {self.size} demo transitions")
    
    def sample(self, batch_size):
        """Sample random batch from demo buffer"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.observations[indices], self.actions[indices]
    
    def get_all_batches(self, batch_size):
        """Get iterator over all demo data"""
        indices = torch.randperm(self.size, device=self.device)
        for i in range(0, self.size, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield self.observations[batch_indices], self.actions[batch_indices]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent with actor-critic architecture"""
    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = np.prod(envs.single_action_space.shape)
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        
        # Actor network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_shape), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_shape) * -0.5)

    def get_value(self, x):
        return self.critic(x)
    
    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def bc_pretrain(agent, demo_buffer, args, device, writer, run_name):
    """Behavior Cloning pretraining phase"""
    print("\n" + "="*50)
    print("PHASE 1: BC PRETRAINING")
    print("="*50)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.bc_lr)
    agent.train()
    
    best_eval_success = 0.0
    
    # Setup evaluation environments
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        # sim_backend="gpu",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    eval_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        video_dir=f"runs/{run_name}/bc_videos" if args.capture_video else None,
    )

    # eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)


    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_envs = RecordEpisode(
            eval_envs, 
            output_dir=f"runs/{run_name}/bc_videos",
            save_trajectory=False,
            max_steps_per_video=200,
            video_fps=30
        )
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True)
    
    for iteration in range(args.bc_iterations):
        # Sample batch from demo buffer
        obs, actions = demo_buffer.sample(args.bc_batch_size)
        
        # Compute BC loss
        pred_actions = agent.actor_mean(obs)
        bc_loss = nn.functional.mse_loss(pred_actions, actions)
        
        # Update
        optimizer.zero_grad()
        bc_loss.backward()
        optimizer.step()
        
        # Logging
        if iteration % 1000 == 0:
            writer.add_scalar("bc/loss", bc_loss.item(), iteration)
            print(f"BC Iteration {iteration}/{args.bc_iterations}, Loss: {bc_loss.item():.4f}")
        
        # Evaluation
        if iteration % args.bc_eval_freq == 0 or iteration == args.bc_iterations - 1:
            agent.eval()
            eval_metrics = evaluate_agent(agent, eval_envs, args.num_eval_episodes, args.num_eval_steps, device)
            agent.train()
            
            for k, v in eval_metrics.items():
                writer.add_scalar(f"bc_eval/{k}", v, iteration)
            
            print(f"BC Eval - Success: {eval_metrics['success_rate']:.3f}, Return: {eval_metrics['return']:.2f}")
            
            # Save best model
            if eval_metrics['success_rate'] > best_eval_success:
                best_eval_success = eval_metrics['success_rate']
                save_checkpoint(agent, run_name, "bc_best")
                print(f"New best BC model! Success rate: {best_eval_success:.3f}")
    
    # Save final BC model
    save_checkpoint(agent, run_name, "bc_final")
    eval_envs.close()
    print(f"BC Pretraining completed. Best success rate: {best_eval_success:.3f}")
    return agent


# def evaluate_agent(agent, eval_envs, num_episodes, num_steps, device):
#     """Evaluate agent and return metrics"""
#     def sample_fn(obs):
#         if isinstance(obs, np.ndarray):
#             obs = torch.from_numpy(obs).float().to(device)
#         with torch.no_grad():
#             action = agent.get_action(obs, deterministic=True)
#         return action
    
#     eval_metrics = bc_evaluate(num_episodes, sample_fn, eval_envs)
    
#     metrics = {}
#     for k, v in eval_metrics.items():
#         metrics[k.replace('_at_end', '_rate') if 'success' in k else k] = np.mean(v)
    
#     if 'r' in metrics:
#         metrics['return'] = metrics.pop('r')
#     if 'l' in metrics:
#         metrics['length'] = metrics.pop('l')
#     if 'success_once' in metrics:
#         metrics['success_rate'] = metrics['success_once']
    
#     return metrics

def evaluate_agent(agent, eval_envs, num_episodes, num_steps, device):
    """Evaluate agent and return metrics - DEBUG VERSION"""
    
    eval_obs, _ = eval_envs.reset()
    
    for step in range(num_steps):
        with torch.no_grad():
            actions = agent.get_action(eval_obs, deterministic=True)
            eval_obs, rewards, terminations, truncations, infos = eval_envs.step(actions)
            
            # 디버깅: 첫 번째 truncation에서 info 출력
            if truncations.any() or terminations.any():
                print("\n" + "="*60)
                print("EPISODE ENDED - Debugging Info Structure")
                print("="*60)
                print(f"terminations: {terminations}")
                print(f"truncations: {truncations}")
                print(f"\ninfo keys: {infos.keys()}")
                print(f"info type: {type(infos)}")
                
                for key in infos.keys():
                    print(f"\n--- infos['{key}'] ---")
                    print(f"  type: {type(infos[key])}")
                    if isinstance(infos[key], dict):
                        print(f"  keys: {infos[key].keys()}")
                        for k, v in infos[key].items():
                            print(f"    {k}: {type(v)} - {v if not torch.is_tensor(v) else v.shape}")
                    elif isinstance(infos[key], (list, tuple)):
                        print(f"  length: {len(infos[key])}")
                        if len(infos[key]) > 0:
                            print(f"  first element type: {type(infos[key][0])}")
                            print(f"  first element: {infos[key][0]}")
                    else:
                        print(f"  value: {infos[key]}")
                
                print("="*60)
                break  # 첫 에피소드 끝나면 멈춤
    
    # 임시로 빈 메트릭 반환
    return {
        "success_rate": 0.0,
        "return": 0.0,
        "length": 0.0,
    }

def save_checkpoint(agent, run_name, tag):
    """Save model checkpoint"""
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save(agent.state_dict(), f"runs/{run_name}/checkpoints/{tag}.pt")


def rl_training(agent, demo_buffer, args, device, writer, run_name):
    """RL fine-tuning phase with demo augmentation and BC regularization"""
    print("\n" + "="*50)
    print("PHASE 2: RL FINE-TUNING")
    print("="*50)
    
    # Setup environments
    env_kwargs = dict(
        control_mode=args.control_mode,
        obs_mode="state",
        render_mode="rgb_array",
        # sim_backend="gpu"
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    
    # envs = gym.make(args.env_id, num_envs=args.num_envs, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    # eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)

    envs = gym.make(args.env_id, num_envs=args.num_envs, reconfiguration_freq=args.reconfiguration_freq, sim_backend="gpu", **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        "gpu",
        env_kwargs,
        video_dir=f"runs/{run_name}/rl_videos" if args.capture_video else None,
    )
    
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    
    if args.capture_video:
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=f"runs/{run_name}/rl_videos",
            save_trajectory=False,
            max_steps_per_video=200,
            video_fps=30
        )
    
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True)
    
    # Setup optimizer
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Storage for rollouts
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
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
    
    action_space_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(envs.single_action_space.high).to(device)
    
    best_eval_success = 0.0
    bc_reg_coef = args.bc_reg_coef
    
    for iteration in range(1, args.rl_iterations + 1):
        print(f"\nRL Iteration {iteration}/{args.rl_iterations}, global_step={global_step}")
        
        # Decay BC regularization coefficient
        if args.use_bc_reg:
            bc_reg_coef *= args.bc_reg_decay
        
        # Evaluation
        if iteration % args.eval_freq == 1:
            agent.eval()
            eval_metrics = evaluate_agent(agent, eval_envs, args.num_eval_episodes, args.num_eval_steps, device)
            agent.train()
            
            for k, v in eval_metrics.items():
                writer.add_scalar(f"rl_eval/{k}", v, global_step)
            
            print(f"RL Eval - Success: {eval_metrics['success_rate']:.3f}, Return: {eval_metrics['return']:.2f}")
            
            if eval_metrics['success_rate'] > best_eval_success:
                best_eval_success = eval_metrics['success_rate']
                save_checkpoint(agent, run_name, "rl_best")
                print(f"New best RL model! Success rate: {best_eval_success:.3f}")
        
        # Collect rollout
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
            
            # Execute action
            next_obs, reward, terminations, truncations, infos = envs.step(
                torch.clamp(action, action_space_low, action_space_high)
            )
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale
            
            # Log training metrics
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    writer.add_scalar(f"rl_train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = \
                        agent.get_value(infos["final_observation"][done_mask]).view(-1)
        
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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Update policy
        agent.train()
        b_inds = np.arange(args.batch_size)
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                # Calculate how many demo samples to include
                demo_size = int(args.minibatch_size * args.demo_buffer_ratio) if args.use_demo_buffer else 0
                online_size = args.minibatch_size - demo_size
                
                # Online data
                mb_online_inds = mb_inds[:online_size]
                
                # Get PPO loss on online data
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_online_inds], b_actions[mb_online_inds]
                )
                logratio = newlogprob - b_logprobs[mb_online_inds]
                ratio = logratio.exp()
                
                # Advantages normalization
                mb_advantages = b_advantages[mb_online_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_online_inds]) ** 2
                    v_clipped = b_values[mb_online_inds] + torch.clamp(
                        newvalue - b_values[mb_online_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_online_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_online_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                
                # BC regularization loss on demo data
                bc_loss = torch.tensor(0.0, device=device)
                if demo_size > 0 and (args.use_bc_reg or args.use_demo_buffer):
                    demo_obs, demo_actions = demo_buffer.sample(demo_size)
                    pred_demo_actions = agent.actor_mean(demo_obs)
                    bc_loss = nn.functional.mse_loss(pred_demo_actions, demo_actions)
                
                # Total loss
                rl_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                
                if args.use_bc_reg:
                    total_loss = rl_loss + bc_reg_coef * bc_loss
                else:
                    total_loss = rl_loss + bc_loss * 0.0  # Still compute but don't use
                
                # Update
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
        
        # Logging
        writer.add_scalar("rl/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("rl/value_loss", v_loss.item(), global_step)
        writer.add_scalar("rl/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("rl/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("rl/bc_loss", bc_loss.item(), global_step)
        writer.add_scalar("rl/bc_reg_coef", bc_reg_coef, global_step)
        writer.add_scalar("rl/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        if iteration % 10 == 0:
            print(f"PG Loss: {pg_loss.item():.4f}, V Loss: {v_loss.item():.4f}, "
                  f"BC Loss: {bc_loss.item():.4f}, BC Coef: {bc_reg_coef:.4f}")
    
    # Save final model
    save_checkpoint(agent, run_name, "rl_final")
    envs.close()
    eval_envs.close()
    
    print(f"\nRL Training completed. Best success rate: {best_eval_success:.3f}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Compute batch sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    # Setup run name
    if args.exp_name is None:
        args.exp_name = "bc2rl"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Initialize wandb
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            group="BC2RL",
            tags=["bc2rl", "demo_augmented", "bc_regularization"],
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Load demo buffer
    demo_buffer = DemoBuffer(args.demo_path, device, load_count=args.num_demos)
    
    # Create temporary env to get observation/action space
    temp_env = gym.make(
        args.env_id,
        num_envs=1,
        control_mode=args.control_mode,
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="gpu"
    )
    if isinstance(temp_env.action_space, gym.spaces.Dict):
        temp_env = FlattenActionSpaceWrapper(temp_env)
    
    # Initialize agent
    agent = Agent(temp_env).to(device)
    temp_env.close()
    
    print(f"\nExperiment: {run_name}")
    print(f"Device: {device}")
    print(f"Environment: {args.env_id}")
    print(f"BC Iterations: {args.bc_iterations}")
    print(f"RL Iterations: {args.rl_iterations}")
    print(f"Demo buffer size: {demo_buffer.size}")
    print(f"Demo buffer ratio: {args.demo_buffer_ratio}")
    print(f"BC reg coefficient: {args.bc_reg_coef} (decay: {args.bc_reg_decay})")
    print(f"Use BC regularization: {args.use_bc_reg}")
    print(f"Use demo buffer: {args.use_demo_buffer}")
    
    # Phase 1: BC Pretraining
    agent = bc_pretrain(agent, demo_buffer, args, device, writer, run_name)
    
    # Phase 2: RL Fine-tuning
    rl_training(agent, demo_buffer, args, device, writer, run_name)
    
    # Cleanup
    writer.close()
    if args.track:
        wandb.finish()
    
    print("\n" + "="*50)
    print("BC2RL Training Complete!")
    print("="*50)
    print(f"Checkpoints saved in: runs/{run_name}/checkpoints/")
    print(f"Videos saved in: runs/{run_name}/bc_videos/ and runs/{run_name}/rl_videos/")
    print(f"Tensorboard logs: runs/{run_name}/")
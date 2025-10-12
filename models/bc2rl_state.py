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
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
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
    max_episode_steps: Optional[int] = 150
    """Change the environments' max_episode_steps"""
    sim_backend: str = "gpu"
    """the simulation backend for environments"""

    # BC pretraining arguments
    demo_path: str = "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5"
    """the path of demo dataset"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    bc_epochs: int = 50
    """number of epochs for BC pretraining"""
    bc_batch_size: int = 512
    """batch size for BC pretraining"""
    bc_lr: float = 1e-3
    """learning rate for BC pretraining"""
    
    # RL training arguments
    total_timesteps: int = 10_000_000
    """total timesteps for RL training"""
    learning_rate: float = 3e-4
    """learning rate for RL training"""
    num_envs: int = 512
    """number of parallel environments"""
    num_steps: int = 50
    """number of steps per rollout"""
    gamma: float = 0.8
    """discount factor"""
    gae_lambda: float = 0.9
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
    bc_reg_coef: float = 0.8
    """BC regularization coefficient for RL training"""
    bc_reg_decay: bool = True
    """whether to decay BC regularization coefficient"""
    bc_reg_min: float = 0.1
    """minimum BC regularization coefficient"""
    bc_reg_decay_steps: int = 10_000_000
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


class BCAgent(nn.Module):
    """Agent for BC pretraining (deterministic policy)"""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # self.net = nn.Sequential(
        #     layer_init(nn.Linear(state_dim, 256)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(256, 256)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(256, 256)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(256, action_dim), std=0.01),
        # )
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PPOAgent(nn.Module):
    """Agent for PPO training with stochastic policy"""
    def __init__(self, envs, bc_agent: Optional[BCAgent] = None):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        
        # Actor network - initialize from BC agent if provided
        if bc_agent is not None:
            self.actor_mean = bc_agent.net
        else:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, act_dim), std=0.01*np.sqrt(2)),
            )
        
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        
        # Store BC agent for regularization
        self.bc_agent = bc_agent
        if self.bc_agent:
            # Freeze BC agent parameters
            for param in self.bc_agent.parameters():
                param.requires_grad = False
    
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
    
    def get_bc_loss(self, obs, actions):
        """Compute BC regularization loss"""
        if self.bc_agent is None:
            return torch.tensor(0.0)
        with torch.no_grad():
            bc_actions = self.bc_agent(obs)
        pred_actions = self.actor_mean(obs)
        return F.mse_loss(pred_actions, bc_actions)


class ManiSkillDataset(Dataset):
    def __init__(self, dataset_file: str, device, load_count=None):
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        
        self.observations = []
        self.actions = []
        
        if load_count is None:
            load_count = len(self.episodes)
        
        print(f"Loading {load_count} episodes for BC pretraining...")
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            
            obs_data = trajectory["obs"][:]
            action_data = trajectory["actions"][:]
            
            self.observations.append(obs_data[:-1])
            self.actions.append(action_data)
        
        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)
        self.device = device
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx]).float().to(self.device)
        action = torch.from_numpy(self.actions[idx]).float().to(self.device)
        return obs, action


def pretrain_bc(args, bc_agent, dataset, device, writer=None):
    """Pretrain agent with behavior cloning"""
    print("\n" + "="*50)
    print("Starting BC Pretraining...")
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
            obs, actions = batch
            pred_actions = bc_agent(obs)
            loss = F.mse_loss(pred_actions, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            global_bc_step += 1

            if batch_idx % 20 == 0 and writer is not None:
                writer.add_scalar("bc_pretrain/loss", loss.item(), global_bc_step)

        
        avg_loss = np.mean(epoch_losses)
        print(f"BC Epoch {epoch+1}/{args.bc_epochs}, Loss: {avg_loss:.4f}")
        if writer is not None:
            writer.add_scalar("bc_pretrain/epoch_loss", avg_loss, epoch+1)


        if (epoch + 1) % 25 == 0 and writer is not None:
            print("Evaluating BC agent...")
            bc_agent.eval()
            
            def sample_fn(obs):
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float().to(device)
                return bc_agent(obs)
            
            with torch.no_grad():
                eval_metrics = evaluate(args.num_eval_episodes, sample_fn, eval_envs)
            
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                if writer is not None:
                    writer.add_scalar(f"bc_pretrain/eval_{k}", eval_metrics[k], epoch+1)
            
            success_rate = eval_metrics.get("success_at_end", 0.0)
            print(f"BC Eval success rate: {success_rate:.4f}")
            
            if success_rate > best_bc_eval_success:
                best_bc_eval_success = success_rate
                save_path = f"runs/{run_name}/best_bc_model.pt"
                torch.save(bc_agent.state_dict(), save_path)
                print(f"New best BC model saved (success {best_bc_eval_success:.4f}): {save_path}")
            
            bc_agent.train()

    
    print("BC Pretraining completed!\n")
    return bc_agent


def train_rl(args, agent, envs, eval_envs, device, writer):
    """Train agent with PPO + BC regularization"""
    print("\n" + "="*50)
    print("Starting RL Training with BC Regularization...")
    print("="*50)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Storage setup
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
            
            # Clip actions to action space bounds
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
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/bc_loss", bc_loss.item(), global_step)
        writer.add_scalar("losses/bc_coef", current_bc_coef, global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        SPS = int(global_step / (time.time() - start_time))
        print(f"Iteration {iteration}/{args.num_iterations}, SPS: {SPS}, BC coef: {current_bc_coef:.4f}")
        writer.add_scalar("charts/SPS", SPS, global_step)
        
        # Evaluation
        if iteration % args.eval_freq == 0:
            print("Evaluating...")
            agent.eval()
            
            def sample_fn(obs):
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float().to(device)
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
        # reward_mode="sparse", #
        obs_mode="state",
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    
    # Training environments
    train_env_kwargs = env_kwargs.copy()
    train_env_kwargs["sim_backend"] = "gpu" if args.sim_backend == "gpu" else "cpu"
    
    envs = gym.make(args.env_id, num_envs=args.num_envs, **train_env_kwargs)
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
    eval_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
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
            group="BC2RL",
            tags=["bc2rl", "bc_pretrain", "ppo_finetune"],
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Load dataset for BC pretraining
    dataset = ManiSkillDataset(args.demo_path, device=device, load_count=args.num_demos)
    
    # Initialize BC agent and pretrain
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    act_dim = np.prod(envs.single_action_space.shape)
    bc_agent = BCAgent(obs_dim, act_dim).to(device)
    bc_agent = pretrain_bc(args, bc_agent, dataset, device, writer)
    
    # Save BC pretrained model
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    bc_save_path = f"runs/{run_name}/bc_pretrained.pt"
    torch.save(bc_agent.state_dict(), bc_save_path)
    print(f"BC pretrained model saved: {bc_save_path}")
    
    # Initialize PPO agent with BC agent
    ppo_agent = PPOAgent(envs, bc_agent).to(device)
    
    # Train with RL + BC regularization
    train_rl(args, ppo_agent, envs, eval_envs, device, writer)
    
    # Cleanup
    envs.close()
    eval_envs.close()
    writer.close()
    
    if args.track:
        wandb.finish()
# rl with bc regularization
# python ./models/bc2rl.py \
#   --env_id PickCube-v1 \
#   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
#   --control-mode "pd_ee_delta_pos" \
#   --sim-backend "gpu" --max-episode-steps 100 \
#   --bc_iterations 30000 \
#   --rl_iterations 3000 \

# bc2rl version2
# python models/bc2rl_regularization.py \
#   --env_id PickCube-v1 \
#   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
#   --control-mode "pd_ee_delta_pos" \
#   --bc_epochs 50 \
#   --total_timesteps 10000000

# bc2rl version2 - sparse reward + epoch 200
# python models/bc2rl_regularization.py \
#   --env_id PickCube-v1 \
#   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
#   --control-mode "pd_ee_delta_pos" \
#   --bc_epochs 200 \
#   --total_timesteps 10000000


# bc2rl state experiments - sparse reward
# python models/bc2rl_regularization.py \
#   --env_id PickCube-v1 \
#   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
#   --control-mode "pd_ee_delta_pos" \
#   --bc_epochs 200 \
#   --total_timesteps 100000000

# python models/bc2rl_regularization.py \
#   --env_id PickCube-v1 \
#   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
#   --control-mode "pd_ee_delta_pos" \
#   --bc_epochs 1000 \
#   --total_timesteps 100000000

# python models/bc2rl_regularization.py \
#   --env_id PickCube-v1 \
#   --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
#   --control-mode "pd_ee_delta_pos" \
#   --bc_epochs 2000 \
#   --total_timesteps 100000000

# python models/bc2rl_state.py \
#   --env_id PushT-v1 \
#   --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cuda.h5 \
#   --control-mode "pd_ee_delta_pos" \
#   --bc_epochs 200 \
#   --max_episode_steps 100 \
#   --total_timesteps 10000000 \
#   --num_envs 32 \
#   --bc_reg_coef 0.3 \
#   --bc_reg_min 0 \
#   --bc_reg_decay_steps 200000 \


python models/bc2rl_state.py \
  --env_id PushT-v1 \
  --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cuda.h5 \
  --control-mode "pd_ee_delta_pos" \
  --bc_epochs 0 \
  --max_episode_steps 100 \
  --total_timesteps 100000000 \
  --num_envs 512 \
  --bc_reg_coef 0 \
  --bc_reg_min 0 \
  --bc_reg_decay_steps 10 \
  --eval_freq 10


python models/bc2rl_state.py \
  --env_id PushT-v1 \
  --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cuda.h5 \
  --control-mode "pd_ee_delta_pos" \
  --bc_epochs 150 \
  --max_episode_steps 100 \
  --total_timesteps 100000000 \
  --num_envs 128 \
  --bc_reg_coef 0.5 \
  --bc_reg_min 0 \
  --bc_reg_decay_steps 5000 \
  --eval_freq 10






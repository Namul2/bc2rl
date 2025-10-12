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

python models/bc2rl_regularization.py \
  --env_id PushT-v1 \
  --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cuda.h5 \
  --control-mode "pd_ee_delta_pos" \
  --bc_epochs 1000 \
  --max_episode_steps 100 \
  --total_timesteps 100000000





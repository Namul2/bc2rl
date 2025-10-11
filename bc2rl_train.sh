# rl with bc regularization
python ./models/bc2rl.py \
  --env_id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" \
  --sim-backend "gpu" --max-episode-steps 100 \
  --bc_iterations 30000 \
  --rl_iterations 3000 \
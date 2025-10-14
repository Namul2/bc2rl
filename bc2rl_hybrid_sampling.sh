# bc2rl hybrid sampling
python models/bc2rl_hybrid_sampling.py \
  --env_id PushT-v1 \
  --control-mode "pd_ee_delta_pos" \
  --bc_checkpoint runs/PushT-v1__bc2rl_regularization__42__1760278051/best_bc_model.pt \
  --initial_bc_prob 0.5 \
  --bc_prob_decay 0.99 \
  --min_bc_prob 0.0 \
  --num_envs 256 \


  # --demo-path ~/.maniskill/demos/PushT-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cuda.h5 \



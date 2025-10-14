# rl with bc regularization rgbd version

CUBLAS_WORKSPACE_CONFIG=:16:8 python ./models/bc2rl_rgbd.py \
    --env_id PushT-v1 \
    --demo_path ~/.maniskill/demos/PushT-v1/rl/trajectory.rgbd.pd_ee_delta_pos.physx_cuda.h5 \
    --control_mode "pd_ee_delta_pos" \

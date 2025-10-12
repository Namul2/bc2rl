# rl with bc regularization rgbd version
python bc2rl_rgb.py \
    --env_id "PegInsertionSide-v0" \
    --demo_path "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.rgbd.pd_ee_delta_pose.h5" \
    --control_mode "pd_ee_delta_pose" \
    --use_depth True \
    --include_state True \
    --bc_epochs 50 \
    --bc_batch_size 64 \
    --num_envs 256 \
    --bc_reg_coef 0.5
# PickCube-v1
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o state \
#   --save-traj --num-envs 10 -b cpu


python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgbd \
  --save-traj --num-envs 10 -b cpu

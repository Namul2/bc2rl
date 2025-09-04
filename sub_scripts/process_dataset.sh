# no.1 problem: 
# for processing datasets especially for difficult ones like PushT-v1
# it fails to replaying the trajectories due to non-deterministic physics sim

# no.1 solution: 
# do not use use-env-states, not sepecifying cpu/gpu device(it automatically uses same device as collected ones)
# still PegInsertionSide-v1 fails, others 99% work fine

# PickCube-v1 --------------------------------------------------
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o state \
#   --save-traj --num-envs 10 -b cpu

# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o rgbd \
#   --save-traj --num-envs 10 -b cpu


# PegInsertionSide-v1 --------------------------------------------------
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o state \
#   --save-traj # --num-envs 10 
  # --use-env-states \

# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o rgbd \
#   --save-traj --num-envs 10 -b cpu


# PushT-v1 --------------------------------------------------
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o state \
#   --save-traj --num-envs 10

# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pos.physx_cuda.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o rgbd \
#   --save-traj --num-envs 10

# StackCube-v1 --------------------------------------------------
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o state \
#   --save-traj --num-envs 10 

# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path ~/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5 \
#   --use-first-env-state -c pd_ee_delta_pos -o rgbd \
#   --save-traj --num-envs 10 

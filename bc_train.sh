# PickCube-v1
python models/bc.py --env-id "PickCube-v1" \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "gpu" --max-episode-steps 100 \
  --total-iters 30000

  
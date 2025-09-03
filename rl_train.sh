# RL only training scripts (baselines)

### State Based PPO ###
# python models/ppo.py --env_id="PickCube-v1" \
#   --num_envs=512 --update_epochs=8 --num_minibatches=32 \
#   --total_timesteps=10_000_000


### RGB Based PPO ###
# python models/ppo_rgb.py --env_id="PickCube-v1" \
#   --num_envs=256 --update_epochs=8 --num_minibatches=8 \
#   --total_timesteps=10_000_000

# python models/ppo_rgb.py --env_id="AssemblingKits-v1" \
#   --num_envs=128 --update_epochs=8 --num_minibatches=8 \
#   --total_timesteps=10_000_000

python models/ppo_rgb.py --env_id="PegInsertionSide-v1" \
  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
  --total_timesteps=10_000_000
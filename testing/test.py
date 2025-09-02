import gymnasium as gym
import mani_skill.envs

# ---- env parameters ----
env_name = "PickCube-v1"
num_envs = 1
obs_mode = "state" # state_dict, rgbd, 
control_mode="pd_joint_delta_pos", # there is also "pd_joint_delta_pos", ...
render_mode = "human" # None, human, rgb_array,
seed = 42

# Create the environment
env = gym.make(
    env_name,
    num_envs=num_envs,
    obs_mode=obs_mode,
    # control_mode=control_mode,
    render_mode=render_mode
)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

obs, _ = env.reset(seed=seed)
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action) # test sampling
    done = terminated or truncated
    env.render()

env.close()
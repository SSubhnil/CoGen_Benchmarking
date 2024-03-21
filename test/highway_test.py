import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
from IPython import display

env = gym.make("highway-v0")
env.configure({"screen_width": 640,
               "screen_height": 400})


obs = env.reset()

plt.figure(figsize=(9, 5))

for _ in range(100):

    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)

    env.render("rgb_array")

    if done:
        obs = env.reset()

env.close()
import gymnasium as gym

env = gym.make("Ant-v4", render_mode='human')

env.reset()

for _ in range(1000):

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()

    if done:
        env.reset()

env.close()


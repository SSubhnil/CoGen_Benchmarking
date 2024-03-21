import gymnasium as gym
from minigrid import minigrid_env
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

def run_episode(env):
    env.reset()
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample()  # Replace this with your agent's policy
        obs, reward, done, info, _ = env.step(action)
        total_reward += reward
        env.render()

    return total_reward


def main():
    # Create the Lava Crossing environment
    env = gym.make('MiniGrid-LavaGapS5-v0')

    # Optionally wrap the environment to simplify observation space
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field

    episodes = 5  # Run 5 episodes for the experiment

    for episode in range(episodes):
        total_reward = run_episode(env)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
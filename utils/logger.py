import imageio
import wandb
import os
from torch.utils.tensorboard import SummaryWriter
from dm_env import Environment
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class TrainingLogger:
    def __init__(self, my_project_name, domain_name, task_name, my_wandb_username, log_mode, config=None):
        self.project_name = my_project_name
        self.domain_name = domain_name
        self.task_name = task_name
        wandb.login(key=my_wandb_username)
        wandb.init(project=my_project_name, mode=log_mode)
        self.config = config
        self.eval_ep_reward = 0
        wandb.define_metric("evaluation/Eval_reward", step_metric="evaluation_step")
        wandb.define_metric("evaluation/Eval_ep_reward", step_metric="evaluation_step")


    @staticmethod
    def log_training(reward, step, episodic=False, ep_num=None):
        """
        :param reward: Log a single reward
        :param step: Step info
        :return: None
        """
        if episodic:
            # Log episodic reward with the episode number
            wandb.log({"Episodic Reward": reward, "Episode": ep_num}, step=step+1)
        else:
            # Log step reward
            wandb.log({"Step Reward": reward}, step=step+1)

    def log_evaluation(self, reward, evaluation_step, dones):

        wandb.log({"evaluation/Eval_reward": reward, "evaluation_step": evaluation_step})
        self.eval_ep_reward += reward
        if dones:
            wandb.log({"evaluation/Eval_ep_reward": self.eval_ep_reward, "evaluation_step": evaluation_step})
            self.eval_ep_reward = 0

class ImageLogger(TrainingLogger):
    def __init__(self, my_project_name, domain_name, task_name, my_wandb_username, log_mode):
        super().__init__(my_project_name, domain_name, task_name, my_wandb_username, log_mode, config=None)
        self.folder_path = os.path.join('results', f'images_{self.project_name}_{self.task_name}')
        os.makedirs(self.folder_path, exist_ok=True)

    @staticmethod
    def capture_image(env, mode='rgb_array'):
        # Checks if the environment is a DummyVecEnv or VecEnv
        if isinstance(env, DummyVecEnv) or hasattr(env, 'venv'):
            # Assuming we're interested in the first sub-env for image capture
            env = env.envs[0]
        # Directly access unwrapped if it's a gym env
        # If the env is further wrapped
        if hasattr(env, 'unwrapped'):
            env = env.unwrapped

        # Handle dm_control environment
        if isinstance(env, Environment):
            if mode == 'rgb_array':
            # Assuming the agent has an attribute 'centre_of_mass' or similar
            	center_x, center_y = env.physics.names.data.xpos['walker',['x', 'y']]
                return env.physics.render(camera_id=0, width=256, height=256, lookat=[center_x, center_y, 0])
            else:
                print("dm_control environment: 'human' mode rendering does not return an image.")
                return None
        # Handle Gymnasium environments
        elif isinstance(env, gym.Env) or hasattr(env, 'render'):
            # Use provided mode for Gym environments.
            try:
                return env.render(mode=mode)
            except NotImplementedError:
                print(f"Gym environment: Rendering mode '{mode}' not supported")
                return None
        else:
            raise ValueError("Unsupported environment type")


    @staticmethod
    def log_image_tensorboard(image, step):
        SummaryWriter.add_image('Environment', image.transpose(2, 0, 1), global_step=step)

    def store_image(self, image, step):
        filename = f"{self.project_name}_{self.task_name}_frame_{step}.png"
        filepath = os.path.join(self.folder_path, filename)
        imageio.imwrite(filepath, image)

    def create_video_and_upload(self, fps=29):
        """
        Creates a video from images stored in a folder and uploads it to WandB.

        Args:
        - folder_path: The path to the folder where images are stored.
        - project_name: The name of the project, used in the video file name.
        - fps: Frames per second for the output video.
        """
        images = []
        for filename in sorted(os.listdir(self.folder_path)):
            if filename.endswith(".png"):
                filepath = os.path.join(self.folder_path, filename)
                images.append(imageio.v2.imread(filepath))

        # Specify the path for the output video
        video_path = os.path.join(self.folder_path, "evaluation_video.mp4")
        imageio.mimsave(video_path, images, format='FFMPEG', fps=fps)

        # Upload to WandB
        wandb.log({"evaluation_video": wandb.Video(video_path, fps=fps, format="mp4")})


class RewardLoggingCallback(BaseCallback):
    def __init__(self, logger, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.tlogger = logger
        self.episodic_reward = 0
        self.episode_num = 0

    def _on_step(self):
        # This method is called on each step of the training
        self.episodic_reward += self.locals['rewards'][0]

        # Log training reward per step
        self.tlogger.log_training(reward=self.locals['rewards'][0], step=self.num_timesteps)

        return True

    def _on_rollout_end(self):
        # Called at the end of each episode
        # Log episodic reward
        self.tlogger.log_training(reward=self.episodic_reward, step=self.num_timesteps, episodic=True, ep_num=self.episode_num)

        # reset episodic reward
        self.episodic_reward = 0
        self.episode_num += 1


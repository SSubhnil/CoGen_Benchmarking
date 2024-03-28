import torch

from utils import dm2gym, logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import numpy as np

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)  # For CUDA support if available

project_name = "PPO_Walker2d"
wandb_key = "576d985d69bfd39f567224809a6a3dd329326993"
mode = "offline" #"offline"
domain_name = "walker"
task_name = "walk"
total_timesteps = 20000

t_logger = logger.TrainingLogger(my_project_name=domain_name, task_name=task_name,
                                 my_wandb_username=wandb_key, log_mode=mode)
im_logger = logger.ImageLogger(my_project_name=domain_name, task_name=task_name,
                               my_wandb_username=wandb_key, log_mode=mode)

env = dm2gym.DMControlWrapperWithForce(domain_name=domain_name, task_name=task_name, seed=SEED, force_magnitude=40)

# Check the environment
check_env(env, warn=True)
# Wrap the environment in a DummyVecEnv for stable-baselines3
vec_env = make_vec_env(lambda: env, n_envs=1, seed=SEED)

# Initialize the model
model = PPO("MlpPolicy", env, seed=SEED, verbose=1,
            learning_rate=3e-4, #3e-4 default
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.2, policy_kwargs={"net_arch":[128, 128]})
reward_logging_callback = logger.RewardLoggingCallback(logger=t_logger)
model.learn(total_timesteps=total_timesteps, callback=reward_logging_callback)
model.save(domain_name)

obs, _ = env.reset(seed=SEED)
step_ = 0
while step_ <= 1000:
    action, _states = model.predict(obs)
    obs, rewards, dones, _, _ = env.step(action)
    t_logger.log_evaluation(rewards, step_, dones)
    image = im_logger.capture_image(env, mode='human')
    im_logger.store_image(image, step_)
    step_ += 1
im_logger.create_video_and_upload()
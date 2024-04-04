import torch

from utils.logger import TrainingLogger, ImageLogger, RewardLoggingCallback
from utils.env_wrappers import make_custom_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)  # For CUDA support if available

project_name = "PPO_Ant-v4"
wandb_key = "576d985d69bfd39f567224809a6a3dd329326993"
mode = "disabled" #"offline"
domain_name = "Ant-v4" # Use v4 for all gymnasium environments
custom_domain_name = "CustomAnt-v4"
task_name = "walk"
total_timesteps = 1000
force_mag = 30.0
force_body_part_id = 1 # torso
force_duration = 10
num_intervals = 5

t_logger = TrainingLogger(my_project_name=project_name, task_name=task_name, my_wandb_username=wandb_key, log_mode=mode)
im_logger = ImageLogger(project_name, task_name, wandb_key, mode)

"""
A custom Gymnasium wrapper to play with the forces and the MuJoCo physics.
Replace with the following for vanilla environments.
    env_id=domain_name
    env = make_vec_env(env_id, seed=SEED)
"""
env = make_vec_env(env_id=custom_domain_name, n_envs=1,
                   env_kwargs={'force_magnitude': force_mag,
                               'force_body_part_id': force_body_part_id,
                               'total_timesteps': total_timesteps,
                               'force_duration': force_duration,
                               'num_intervals': num_intervals,
                               'seed': SEED})


# Initialize the model
model = PPO("MlpPolicy", env, seed=SEED, verbose=1,
            learning_rate=3e-4,  # 3e-4 default
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.2, policy_kwargs={"net_arch":[128, 128]})
reward_logging_callback = RewardLoggingCallback(logger=t_logger)
model.learn(total_timesteps=total_timesteps, callback=reward_logging_callback)
model.save(domain_name)

obs = env.reset()
step_ = 0
while step_ <= 200:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _ = env.step(action)
    t_logger.log_eval(rewards, step_, dones)
    image = im_logger.capture_image(env)
    im_logger.store_image(image, step_)
    step_ += 1
im_logger.create_video_and_upload()
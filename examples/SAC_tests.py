import torch

from utils import dm2gym, logger
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import numpy as np

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)  # For CUDA support if available

project_name = "SAC_Walker2d"
wandb_key = "576d985d69bfd39f567224809a6a3dd329326993"
mode = "offline" #"offline"
domain_name = "walker"
task_name = "walk"
wind_type = "swelling" # Can be "step" or "swelling"
total_timesteps = 2000000

t_logger = logger.TrainingLogger(my_project_name=project_name, domain_name=domain_name, task_name=task_name,
                                 my_wandb_username=wandb_key, log_mode=mode)
im_logger = logger.ImageLogger(my_project_name=project_name,domain_name=domain_name, task_name=task_name,
                               my_wandb_username=wandb_key, log_mode=mode)

env = dm2gym.DMControlWrapper(domain_name=domain_name, task_name=task_name, seed=SEED) #wind_type=wind_type)

# Check the environment
check_env(env, warn=True)

# Initialize the model
model = SAC("MlpPolicy", env, seed=SEED, verbose=1,
            learning_rate=3e-4, #3e-4 default
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs={"net_arch":[256, 256]})
reward_logging_callback = logger.RewardLoggingCallback(logger=t_logger)
model.learn(total_timesteps=total_timesteps, log_interval=4, callback=reward_logging_callback)

obs, _ = env.reset(seed=SEED)
step_ = 0
while step_ <= 10000:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _, _ = env.step(action)
    t_logger.log_evaluation(rewards, step_, dones)
    image = im_logger.capture_image(env, mode='rgb_array')
    im_logger.store_image(image, step_)
    step_ += 1
im_logger.create_video_and_upload()
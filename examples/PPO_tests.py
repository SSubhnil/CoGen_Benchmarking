from utils import dm2gym, imgproc
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

project_name = "PPO_Walker2d"
wandb_key = "576d985d69bfd39f567224809a6a3dd329326993"

im_store = imgproc.ImageLogger(project_name, wandb_key)

env = dm2gym.DMControlWrapperWithForce(domain_name="walker", task_name="walk")

# Check the environment
check_env(env, warn=True)
# Wrap the environment in a DummyVecEnv for stable-baselines3
vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize the model
model = PPO("MlpPolicy", env, verbose = 1)
model.learn(total_timesteps=100000)
model.save("walker")

del model # remove to demonstrate saving and loading

model = PPO.load("walker")

obs, _ = env.reset()
step_ = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, _, _ = env.step(action)

    image = im_store.capture_image(env)
    imgproc.log_image_wandb(image, step_)
    step_ += 1
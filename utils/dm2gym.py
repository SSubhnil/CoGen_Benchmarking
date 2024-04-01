import gymnasium as gym
import numpy as np
from dm_control import suite
from gymnasium import spaces
from dm_control import viewer
class DMControlWrapper(gym.Env):
    def __init__(self, domain_name, task_name, seed):
        super().__init__()
        # Load the given domain and task from DM Control Suite
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})
        # Extract the action and observation spec from the environment
        if seed is not None:
            self.seed(seed)

        action_spec = self.env.action_spec()
        observation_spec = self.env.observation_spec()
        np.random.seed(seed)
        # Define the action_space and observation_space using Gym spaces
        self.action_space = spaces.Box(low=action_spec.minimum, high=action_spec.maximum, dtype=np.float32)

        # Observation space is a bit more complex due to possible observation dicts
        obs_dim = sum(np.prod(observation_spec[key].shape).astype(int) for key in observation_spec) # Casting into int
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        time_step = self.env.step(action)
        observation = self._flatten_observation(time_step.observation)
        reward = time_step.reward if time_step.reward is not None else 0
        done = time_step.last()
        truncated = False
        info = {}
        return observation, reward, done, truncated, info

    def reset(self, seed=None, **kwargs):
        # Optionally, you can handle the seed here if applicable.
        # For DM Control Suite, seeding is not straightforward as it uses its own RNG.
        # This placeholder allows compatibility with Gym's interface.
        time_step = self.env.reset()
        initial_observation = self._flatten_observation(time_step.observation)

        # Create an empty info dictionary
        info = {}

        return initial_observation, info

    @property
    def physics(self):
        return self.env.physics

    def render(self, mode='human', width=640, height=480):
        if mode == 'rgb_array':
            # Use dm_control's rendering for offscreen rendering
            return self.env.physics.render(width=width, height=480)
        elif mode == 'human':
            # Handle human mode rendering if needed
            # This might involve on-screen rendering, which dm_control environments handle auto
            print("Human mode rendering is not directly supported in this wrapper.")
            # viewer.launch(self.env)
        else:
            raise NotImplementedError(f"{mode} rendering mode is not supported by this wrapper.")

    def close(self):
        # DM Control environments do not require explicit closure methods -> safety
        pass

    def _flatten_observation(self, observation):
        "Flatten observation into a single numpy array."
        flat_obs = np.concatenate([observation[key].ravel() for key in observation])
        return flat_obs.astype(np.float32) # Cast to float32

    def seed(self, seed=None):
        self.env.task.random.seed(seed)

class DMControlWrapperWithForce(DMControlWrapper):
    def __init__(self, domain_name, task_name, seed, force_magnitude = 50, apply_force_steps=1, *args, **kwargs):
        super().__init__(domain_name, task_name,seed, *args, **kwargs)
        self.force_magnitude = force_magnitude # The magnitude of the unbalancing force
        self.apply_force_steps = apply_force_steps # Probability of applying the force at each timestep
        self.step_counter = 0

    def apply_unbalancing_force(self, physics):
        "Applies an unbalancing force to the walker randomly in the left or right direction."
        # Randomly choose a direction for the force; for example, left (-1) or right (1) on the x-axis
        body_part = 'torso'
        body_id = physics.model.name2id(body_part, 'body')
        # Force is applied in the leftward direction (-ve x-axis)
        # Force is passed as [x, y, z, torque_x, torque_y, torque_z]
        force = np.array([-self.force_magnitude, 0, 0, 0, 0, 0])
        # physics.apply_force(force, body_id, global_coordinate=True)
        physics.data.xfrc_applied[body_id] = force

    def step(self, action):
        self.step_counter += 1
        # if self.step_counter % self.apply_force_steps == 0:
        self.apply_unbalancing_force(self.env.physics)
        return super().step(action)
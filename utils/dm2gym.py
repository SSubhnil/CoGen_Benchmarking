import gymnasium as gym
import numpy as np
from dm_control import suite
from gymnasium import spaces
from dm_control import viewer

class DMControlWrapper(gym.Env):
    def __init__(self, domain_name, task_name):
        super().__init__()
        # Load the given domain and task from DM Control Suite
        self.env = suite.load(domain_name=domain_name, task_name=task_name)

        # Extract the action and observation spec from the environment
        action_spec = self.env.action_spec()
        observation_spec = self.env.observation_spec()

        # Define the action_space and observation_space using Gym spaces
        self.action_space = spaces.Box(low=action_spec.minimum, high=action_spec.maximum, dtype=np.float32)

        # Observation space is a bit more complex due to possible observation dicts
        obs_dim = sum(np.prod(observation_spec[key].shape).astype(int) for key in observation_spec) # Casting into int
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        time_step = self.env.step(action)
        observation = self._flatten_observation(time_step.observation)
        reward = time_step.reward or 0
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

    def render(self, mode='human'):
        if mode == 'human':
            viewer.launch(self.env)
        else:
            raise NotImplementedError("Only human mode is supported for rendering.")

    def close(self):
        # DM Control environments do not require explicit closure methods -> safety
        pass

    def _flatten_observation(self, observation):
        "Flatten observation into a single numpy array."
        flat_obs = np.concatenate([observation[key].ravel() for key in observation])
        return flat_obs.astype(np.float32) # Cast to float32

    def seed(self, seed=None):
        # DM Control Suite doesn't provide a direct way to set the seed.
        pass

class DMControlWrapperWithForce(DMControlWrapper):
    def __init__(self, domain_name, task_name, force_magnitude = 10, apply_force_steps=100, *args, **kwargs):
        super().__init__(self, domain_name, task_name, *args, **kwargs)
        self.force_magnitude = force_magnitude # The magnitude of the unbalancing force
        self.apply_force_steps = apply_force_steps # Probability of applying the force at each timestep
        self.step_counter = 0

    def apply_unbalancing_force(self, physics):
        "Applies an unbalancing force to the walker randomly in the left or right direction."
        # Randomly choose a direction for the force; for example, left (-1) or right (1) on the x-axis
        body_part = 'torso'
        body_id = physics.model.body_name2id(body_part)
        # Force is applied in the leftward direction (-ve x-axis)
        force = np.array([-self.force_magnitude, 0, 0])
        physics.apply_force(force, body_id, global_coordinate=True)

    def step(self, action):
        self.step_counter += 1
        if self.step_counter % self.apply_force_steps == 0:
            self.apply_unbalancing_force(self.env.physics)
        observation, reward, done, info = super().step(action)
        # Now proceed with the regular step function
        return observation, reward, done, info
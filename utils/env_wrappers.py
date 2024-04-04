from typing import Union, Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, make


class UnbalanceForceWrapper(gym.Wrapper):
    def __init__(self, env, force_magnitude=20.0, force_body_part_id=1,
                                        total_timesteps=100000, force_duration=10,
                                        num_intervals=10, seed=42):
        super().__init__(env)
        self.seed = seed
        np.random.seed(self.seed)
        self.force_magnitude = force_magnitude
        self.force_body_part_id = force_body_part_id
        self.total_timesteps = total_timesteps
        self.force_duration = force_duration
        self.num_intervals = num_intervals
        self.force_intervals = self.generate_force_intervals(self.total_timesteps,
                                                             self.num_intervals,
                                                             self.force_duration)
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        if any(start <= self.current_step <= start + self.force_duration for start in self.force_intervals):
            self.apply_unbalance_force()
        return self.env.step(action)

    def apply_unbalance_force(self):
        # Assuming the 'torso' is the body part to apply force
        force = np.array([0, self.force_magnitude, 0, 0, 0, 0])  # Example force vector
        try:
            self.env.unwrapped.sim.data.xfrc_applied[self.force_body_part_id] = force
        except AttributeError:
            pass

    @staticmethod
    def generate_force_intervals(total_timesteps, num_intervals, force_duration):
        # Generates random start points for force application
        interval_starts = np.random.choice(range(total_timesteps - force_duration), num_intervals, replace=False)
        return interval_starts

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

def make_custom_env(domain_name, force_magnitude=20.0,
                    force_body_part_id=1, total_timesteps=100000,
                    force_duration=10, num_intervals=10, seed=42):

    env = gym.make(domain_name)
    wrapped_env = UnbalanceForceWrapper(env, force_magnitude=20.0, force_body_part_id=1,
                                        total_timesteps=100000, force_duration=10,
                                        num_intervals=10, seed=42)
    return wrapped_env


register(id='CustomAnt-v4', entry_point='utils.env_wrappers:make_custom_env')

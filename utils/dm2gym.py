import gymnasium as gym
import numpy as np
from dm_control import suite
from gymnasium import spaces
from dm_control import viewer
class DMControlWrapper(gym.Env):
    def __init__(self, domain_name, task_name, seed, wind_type):
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

        self.wind_active = wind_type
        self.confounder_params = {
            'force_type': wind_type,
            'timing': 'random',
            'body_part': 'torso',
            'random_chance': 0.8,  # Chance to apply random force
            'force_range': (90, 170),
            'interval_mean': 90,  # Mean for sampling interval 90, 180
            'interval_std': 10,  # Standard deviation for sampling interval
            'duration_min': 5,  # Minimum duration for swelling force
            'duration_max': 20  # Maximum duration for the swelling force
        }

        # Initialize attributes based on confounder_params
        self.force_type = self.confounder_params['force_type']
        self.timing = self.confounder_params['timing']
        self.body_part = self.confounder_params['body_part']
        self.random_chance = self.confounder_params['random_chance']
        self.force_range = self.confounder_params['force_range']
        self.interval_mean = self.confounder_params['interval_mean']
        self.interval_std = self.confounder_params['interval_std']
        self.duration_min = self.confounder_params['duration_min']
        self.duration_max = self.confounder_params['duration_max']
        self.time_since_last_force = 0

    def step(self, action):
        # Apply force if enabled
        if self.wind_active is not None:
            self._apply_force()
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

    def _apply_force(self):
        if self.timing == 'random':
            self.interval = max(30, int(np.random.normal(self.interval_mean,
                                                         self.interval_std)))
            if np.random.uniform() > self.random_chance:
                return

        # Update the timing
        self.time_since_last_force += 1
        if self.time_since_last_force < self.interval:
            return

        # Reset timing for next force application
        self.time_since_last_force = 0

        # Sample the force magnitude fom a normal distribution within the range
        force_magnitude = np.clip(np.random.normal((self.force_range[0] + self.force_range[1]) / 2,
                                                   (self.force_range[1] - self.force_range[0]) / 6),
                                  self.force_range[0], self.force_range[1])

        # Calculate the duration for the force application if 'swelling'
        duration = np.random.randint(self.duration_min, self.duration_max + 1)

        # FLipping the direction for additional challenge
        direction = np.random.choice([-1, 1])

        # Apply swelling or other dynamics based on force type
        # Construct the force vector
        if self.force_type == 'step':
            force = np.array([direction * force_magnitude, 0, 0, 0, 0, 0])
        elif self.force_type == 'swelling':
            # Calculate the time step where the force magnitude is at its peak
            peak_time = duration / 2
            # Calculate the standard deviation to control thh width of the bell curve
            sigma = duration / 6  # Adjust as needed for the desired width
            # Calculate the force magnitude at the current time step using a Gaussian function
            time_step_normalized = (self.time_since_last_force - peak_time) / sigma
            magnitude = force_magnitude * np.exp(-0.5 * (time_step_normalized ** 2))
            force = np.array([direction * magnitude, 0, 0, 0, 0, 0])

        body_id = self._env.physics.model.name2id(self.body_part, 'body')
        # Apply the force
        self._env.physics.data.xfrc_applied[body_id] = force
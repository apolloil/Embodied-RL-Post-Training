import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import gym
import yaml
from gym import spaces
import dm_env
from dm_env import StepType, specs, TimeStep
import numpy as np
import cv2
import random
import metaworld
import mujoco
from typing import Any, NamedTuple
from pathlib import Path

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """Wraps a MetaWorld environment to use RGB images as observations.

    Based on BAKU/baku/suite/metaworld.py RGBArrayAsObservationWrapper.
    Modifications: proprioception removed; task_id returned directly;
    HWC -> CHW conversion handled manually.
    """

    def __init__(
        self,
        env,
        task_id,
        ml1,
        width=84,
        height=84,
        max_path_length=500,
    ):
        self._env = env
        self._task_id = task_id
        self.ml1 = ml1
        self._width = width
        self._height = height
        self.max_path_length = max_path_length

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, height, width), dtype=np.uint8
        )
        
        self.action_space = self._env.action_space

        # Action Spec
        wrapped_action_spec = self.action_space
        if not hasattr(wrapped_action_spec, "minimum"):
            wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
        if not hasattr(wrapped_action_spec, "maximum"):
            wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
        
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            np.float32,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

        # Observation Spec
        self._obs_spec = {}
        self._obs_spec["pixels"] = specs.BoundedArray(
            shape=self.observation_space.shape,
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels",
        )
        self._obs_spec["task_id"] = specs.Array(
            shape=(1,),
            dtype=np.int64,
            name="task_id"
        )

    def reset(self, **kwargs):
        """Reset the environment with a randomly sampled task goal."""
        # MetaWorld-specific: randomly sample a task goal before each reset
        task = random.choice(self.ml1.train_tasks)
        self._env.set_task(task)
        self._env._partially_observable = False  # set_task resets this; force goal-observable
        
        self.episode_step = 0
        
        # Call the underlying environment reset
        reset_result = self._env.reset(**kwargs)
        # Gymnasium returns (obs, info); older gym returns obs only
        raw_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        assert raw_obs.shape == (39,), f"Raw observation shape is {raw_obs.shape}, but should be (39,)"

        obs = {}
        # Capture frame and convert HWC -> CHW
        frame = self.get_frame()
        obs["pixels"] = frame.transpose(2, 0, 1).copy()
        obs["task_id"] = np.array([self._task_id], dtype=np.int64)
        obs["goal_achieved"] = False
        # Raw MetaWorld state vector (39-dim float64) for state-based agents
        obs["global_state"] = np.asarray(raw_obs, dtype=np.float32)
        
        return obs

    def step(self, action):
        """Execute one environment step."""
        raw_obs, reward, _, done, info = self._env.step(action)
        
        obs = {}
        # Capture frame and convert HWC -> CHW
        frame = self.get_frame()
        obs["pixels"] = frame.transpose(2, 0, 1).copy() #(C,H,W) uint8
        obs["task_id"] = np.array([self._task_id], dtype=np.int64)
        
        obs["goal_achieved"] = info["success"]
        # Raw MetaWorld state vector (39-dim float64) for state-based agents
        obs["global_state"] = np.asarray(raw_obs, dtype=np.float32)

        self.episode_step += 1
        terminated = bool(info["success"])
        truncated = (self.episode_step == self.max_path_length)
        done = terminated or truncated
        obs["terminated"] = terminated

        return obs, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec
 
    def render(self, mode="rgb_array", width=256, height=256):
        if mode == "rgb_array":
            frame = self.get_frame(width, height)
            return frame
        else:
            self._env.render_mode = "human"
            self._env.render()

    def get_frame(self, width=None, height=None):
        width = self._width if width is None else width
        height = self._height if height is None else height
        frame = self._env.render()[::-1]
        frame = cv2.resize(frame, (width, height))
        return frame # (H, W, 3)

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionDTypeWrapper(dm_env.Environment):
    """Casts actions to the dtype expected by the wrapped environment."""

    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

        # Action spec
        wrapped_action_spec = env.action_space
        if not hasattr(wrapped_action_spec, "minimum"):
            wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
        if not hasattr(wrapped_action_spec, "maximum"):
            wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            np.float32,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )
        self._obs_spec = env.observation_spec()

    def step(self, action):
        action = action.astype(self._env.action_space.dtype)
        # Build a dm_env TimeStep from the raw gym step output
        observation, reward, done, info = self._env.step(action)
        # Action dimension augmentation (+1) is handled by DimensionWrapper, not here
        step_type = StepType.LAST if done else StepType.MID
        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        obs = self._env.reset()
        return TimeStep(
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionRepeatWrapper(dm_env.Environment):
    """Repeats each action for a fixed number of steps, accumulating rewards."""

    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)
    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStepWrapper(dm_env.Environment):
    """Wraps dm_env TimeSteps into ExtendedTimeStep namedtuples with action."""

    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self, time_step, observation=None, action=None, reward=None, discount=None
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


# =============================================================================
# 6. Make Function (Environment Construction)
# =============================================================================
def make(
    task_names,
    cameras,
    img_size,
    action_repeat,
    seed,
    max_episode_steps
):
    """Create a list of wrapped MetaWorld environments for the given tasks.

    Args:
        task_names: List of MetaWorld task name strings.
        cameras: Dict mapping task name to camera name.
        img_size: Observation image size (height and width).
        action_repeat: Number of times each action is repeated.
        seed: Random seed for environment initialization.
        max_episode_steps: Dict mapping task name to max episode length.

    Returns:
        List of fully wrapped dm_env-compatible environments.
    """
    # Use path relative to this script to locate the config file reliably
    script_dir = Path(__file__).parent.parent 
    task_palette_path = script_dir / "conf" / "task_palette.yaml"

    with open(task_palette_path, 'r') as f:
        task_map = yaml.safe_load(f)
    envs = []
    
    # Iterate over the task list and create an environment for each task
    for idx, task_name in enumerate(task_names):
        if task_name not in task_map:
             raise ValueError(f"Task '{task_name}' not found in task_palette.yaml")
        global_task_id = task_map[task_name]  # Global task ID

        # 1. Create the base environment
        ml1 = metaworld.ML1(task_name)
        env = ml1.train_classes[task_name](render_mode="rgb_array")
        env._partially_observable = False  # Goal-observable (aligned with BRC JAX)
        env.seed(seed)
        
        # 2. Set up the camera
        camera_name = cameras[task_name]
        env.camera_name = camera_name
        
        # Ensure the camera is correctly bound
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, env.camera_name)
        env.mujoco_renderer.camera_id = camera_id

        # 3. Apply wrappers
        
        # A. Base wrapper (handles HWC -> CHW conversion)
        env = RGBArrayAsObservationWrapper(
            env,
            task_id=global_task_id,
            ml1=ml1,
            width=img_size,
            height=img_size,
            max_path_length=max_episode_steps[task_name]
        )
        
        # B. Action dtype casting
        env = ActionDTypeWrapper(env, np.float32)
        
        # C. Action repeat
        env = ActionRepeatWrapper(env, action_repeat)
        
        # D. Extended TimeStep wrapper
        env = ExtendedTimeStepWrapper(env)

        envs.append(env)

    return envs

import time
import random
import numpy as np
import torch
import cv2
import imageio
from pathlib import Path

# ==============================================================================
# Utils (adapted from BAKU utils.py)
# ==============================================================================

class eval_mode:
    """Context manager that switches models to eval mode and restores the
    previous training state on exit.

    Supports multiple models simultaneously (e.g. encoder, actor).
    """
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.eval()

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)

def set_seed_everywhere(seed):
    """Set random seeds across all libraries for reproducibility.

    Args:
        seed: Integer seed value.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class Timer:
    """Simple wall-clock timer for measuring training duration."""
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        t = time.time()
        self._start_time = t
        self._last_time = t

    def total_time(self):
        return time.time() - self._start_time

class Until:
    """Callable that returns True while the current step is below the limit.

    Args:
        until_step: Maximum step (exclusive). None means unlimited.

    Example::

        until = Until(1000)
        if until(current_step):
            continue
    """
    def __init__(self, until_step):
        self._until_step = until_step

    def __call__(self, step):
        if self._until_step is None:
            return True
        return step < self._until_step

class Every:
    """Callable that returns True every *every_step* steps.

    Useful for periodic actions such as logging, evaluation, and saving.
    Returns False if *every_step* is None or non-positive.

    Args:
        every_step: Interval in steps.

    Example::

        every = Every(1000)
        if every(step):
            save_snapshot()
    """
    def __init__(self, every_step):
        self._every_step = every_step

    def __call__(self, step):
        if self._every_step is None or self._every_step <= 0:
            return False
        return step > 0 and step % self._every_step == 0

# ==============================================================================
# Video Recorder (adapted from BAKU video.py)
# ==============================================================================

class VideoRecorder:
    """Records environment rendering frames during evaluation and saves them
    as local MP4 files.

    Simplified version without wandb dependency.

    Args:
        root_dir: Root directory; videos are saved under ``root_dir/video/``.
        render_size: Height and width of each rendered frame.
        fps: Frames per second for the output video.
    """
    def __init__(self, root_dir, render_size=224, fps=20):
        self.save_dir = Path(root_dir) / 'video'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, enabled=True):
        """Reset the frame buffer at the start of each episode.

        Args:
            enabled: Whether recording is active.
        """
        self.frames = []
        self.enabled = self.save_dir and enabled

    def record(self, env):
        """Capture a frame from the environment via ``env.render()``.

        Assumes ``env.render()`` returns an ``(H, W, 3)`` numpy array.

        Args:
            env: Environment instance with a ``render`` method.
        """
        if self.enabled:
            # Handle common render interfaces to obtain an RGB array
            if hasattr(env, 'physics'):  # dm_control / MetaWorld (MuJoCo)
                frame = env.render(
                    mode='rgb_array', 
                    height=self.render_size, 
                    width=self.render_size
                )
            else:  # standard Gym
                frame = env.render()

            # Validate and store the frame
            if frame is not None:
                self.frames.append(frame)

    def save(self, file_name):
        """Save buffered frames as an MP4 file using imageio.

        Args:
            file_name: Output filename (e.g. ``'eval_step1000.mp4'``).
        """
        if self.enabled and len(self.frames) > 0:
            path = self.save_dir / file_name
            # imageio expects uint8 data with shape (H, W, 3)
            imageio.mimsave(str(path), self.frames, fps=self.fps)
            
            # Clear buffer to free memory
            self.frames = []

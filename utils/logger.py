import csv
import json
import os
import shutil
from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from termcolor import colored

# Common formatting templates
COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('duration', 'D', 'time'),
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger:
    def __init__(self, log_dir, use_tb=True):
        self._log_dir = log_dir
        self._use_tb = use_tb
        
        # Create directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 1. TensorBoard Writer
        if self._use_tb:
            self._sw = SummaryWriter(log_dir=log_dir)
        
        # 2. Internal cache for aggregating multiple logs within the same step
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, step, n=1):
        """Log a single key-value pair.

        Args:
            key: Metric key, e.g. 'train/loss'.
            value: Scalar value to log.
            step: Current training step.
            n: Sample count for weighted averaging.
        """
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._meters[key].update(value, n)
        
        # Write to TensorBoard in real time (also written collectively at dump)
        if self._use_tb:
            self._sw.add_scalar(key, value, step)

    def log_metrics(self, metrics, step, prefix):
        """Log a dictionary of metrics in batch.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Current training step.
            prefix: Key prefix, e.g. 'train' or 'eval'.
        """
        for key, value in metrics.items():
            self.log(f'{prefix}/{key}', value, step)

    def dump(self, step, save=True, Ty=None):
        """Print cached metrics to the terminal and clear the cache.

        Args:
            step: Current training step.
            save: Whether to persist (unused, kept for API compatibility).
            Ty: Log type, 'train' or 'eval', for colored output.
        """
        if len(self._meters) == 0:
            return

        # Print header
        if Ty == 'train':
            print(colored(f'Step {step} | Training', 'yellow'))
        elif Ty == 'eval':
            print(colored(f'Step {step} | Evaluation', 'green'))
        else:
            print(f'Step {step}')

        # Print detailed metrics
        for key, meter in self._meters.items():
            # Strip the prefix for cleaner display (train/loss -> loss)
            short_key = key.split('/', 1)[-1] 
            print(f'  {short_key:<15} {meter.avg:.4f}')

        # Clear meters for the next cycle
        self._meters.clear()

    def close(self):
        if self._use_tb:
            self._sw.close()

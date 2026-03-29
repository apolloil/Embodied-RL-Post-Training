import torch
import numpy as np


# =====================================================================
# Parallel (per-task) Replay Buffers
# =====================================================================

class ParallelFeatureReplayBuffer:
    """
    Per-task Feature Replay Buffer with (num_tasks, capacity_per_task, ...)
    structure, independent circular pointers, and task-balanced chunk sampling.
    """
    def __init__(self, num_tasks, capacity_per_task, repr_dim, act_dim,
                 device, global_task_ids):
        self.num_tasks = num_tasks
        self.capacity_per_task = capacity_per_task
        self.device = device

        self._global_ids = np.array(global_task_ids, dtype=np.int64)
        self._g2l = {}
        for local, gid in enumerate(self._global_ids):
            self._g2l[int(gid)] = local

        self.ptrs  = np.zeros(num_tasks, dtype=np.int64)
        self.sizes = np.zeros(num_tasks, dtype=np.int64)

        C = capacity_per_task
        self.reprs      = torch.zeros((num_tasks, C, repr_dim), dtype=torch.float32)
        self.next_reprs = torch.zeros((num_tasks, C, repr_dim), dtype=torch.float32)
        self.actions    = torch.zeros((num_tasks, C, act_dim),   dtype=torch.float32)
        self.rewards    = torch.zeros((num_tasks, C, 1),         dtype=torch.float32)
        self.dones      = torch.zeros((num_tasks, C, 1),         dtype=torch.bool)

        self._dones_flat = [self.dones[t].view(-1) for t in range(num_tasks)]

        self._cache_key    = None
        self._gamma_vec    = None
        self._offsets       = None
        self._chunk_arange = None

    # --- properties -----------------------------------------------------------

    @property
    def total_size(self):
        return int(self.sizes.sum())

    @property
    def size(self):
        return self.total_size

    # --- add ------------------------------------------------------------------

    def add(self, repr, action, reward, next_repr, done, task_id):
        t = self._g2l[int(task_id)]
        p = int(self.ptrs[t])

        if isinstance(repr, np.ndarray):      repr      = torch.from_numpy(repr)
        if isinstance(action, np.ndarray):    action    = torch.from_numpy(action)
        if isinstance(next_repr, np.ndarray): next_repr = torch.from_numpy(next_repr)

        self.reprs[t, p]      = repr.flatten()
        self.next_reprs[t, p] = next_repr.flatten()
        self.actions[t, p]    = action.flatten()
        self.rewards[t, p]    = float(reward)
        self.dones[t, p]      = bool(done)

        self.ptrs[t]  = (p + 1) % self.capacity_per_task
        self.sizes[t] = min(self.sizes[t] + 1, self.capacity_per_task)

    # --- cache ----------------------------------------------------------------

    def _ensure_cache(self, chunk_size, gamma):
        key = (chunk_size, gamma)
        if self._cache_key != key:
            self._cache_key    = key
            self._offsets       = torch.arange(chunk_size - 1)
            self._chunk_arange = torch.arange(chunk_size)
            self._gamma_vec    = (
                gamma ** torch.arange(chunk_size, dtype=torch.float32)
            ).view(1, chunk_size, 1)

    # --- per-task filter ------------------------------------------------------

    def _filter_task(self, cands, chunk_size, t):
        df = self._dones_flat[t]
        if chunk_size > 1:
            check = cands.unsqueeze(1) + self._offsets
            bad = df[check].any(dim=1)
        else:
            bad = torch.zeros(len(cands), dtype=torch.bool)

        if self.sizes[t] == self.capacity_per_task:
            ptr = int(self.ptrs[t])
            ptr_bad = (cands < ptr) & (ptr < cands + chunk_size)
            return cands[~bad & ~ptr_bad]
        return cands[~bad]

    # --- sample ---------------------------------------------------------------

    def sample_chunk(self, batch_size, chunk_size, gamma):
        self._ensure_cache(chunk_size, gamma)

        sampleable = [t for t in range(self.num_tasks)
                      if self.sizes[t] >= chunk_size]
        ns = len(sampleable)
        if ns == 0:
            raise RuntimeError("No task has enough data for chunk sampling")

        per_task  = batch_size // ns
        remainder = batch_size % ns
        extra = set(np.random.choice(ns, remainder, replace=False).tolist()) \
                if remainder else set()

        parts_r, parts_nr, parts_a, parts_rew, parts_d, parts_tid = \
            [], [], [], [], [], []

        for i, t in enumerate(sampleable):
            n = per_task + (1 if i in extra else 0)
            if n == 0:
                continue
            mx = int(self.sizes[t]) - chunk_size
            if mx <= 0:
                continue

            cands = torch.from_numpy(
                np.random.randint(0, mx, size=n * 3, dtype=np.int64))
            idx = self._filter_task(cands, chunk_size, t)

            extra_n = n * 8
            while idx.shape[0] < n:
                ec = torch.from_numpy(
                    np.random.randint(0, mx, size=extra_n, dtype=np.int64))
                idx = torch.cat([idx, self._filter_task(ec, chunk_size, t)])
                extra_n *= 2
            idx = idx[:n]

            fin  = idx + chunk_size - 1
            grid = idx.unsqueeze(1) + self._chunk_arange

            parts_r.append(self.reprs[t][idx])
            parts_nr.append(self.next_reprs[t][fin])
            parts_a.append(self.actions[t][grid].reshape(n, -1))
            rseq = self.rewards[t][grid]
            parts_rew.append((rseq * self._gamma_vec).sum(dim=1))
            parts_d.append(self.dones[t][fin].float())
            parts_tid.append(
                torch.full((n,), self._global_ids[t], dtype=torch.long))

        reprs    = torch.cat(parts_r)
        nr       = torch.cat(parts_nr)
        acts     = torch.cat(parts_a)
        rews     = torch.cat(parts_rew)
        dones    = torch.cat(parts_d)
        tids     = torch.cat(parts_tid)

        perm = torch.randperm(tids.shape[0])
        return (
            reprs[perm].to(self.device, non_blocking=True),
            acts[perm].to(self.device, non_blocking=True),
            rews[perm].to(self.device, non_blocking=True),
            nr[perm].to(self.device, non_blocking=True),
            dones[perm].to(self.device, non_blocking=True),
            tids[perm].to(self.device, non_blocking=True),
        )

    # --- persistence ----------------------------------------------------------

    def save(self, path):
        torch.save({
            'reprs': self.reprs, 'next_reprs': self.next_reprs,
            'actions': self.actions, 'rewards': self.rewards,
            'dones': self.dones,
            'ptrs': self.ptrs, 'sizes': self.sizes,
            'global_ids': self._global_ids,
        }, path)

    def load(self, path):
        d = torch.load(path)
        self.reprs      = d['reprs']
        self.next_reprs = d['next_reprs']
        self.actions    = d['actions']
        self.rewards    = d['rewards']
        self.dones      = d['dones']
        self.ptrs       = d['ptrs']
        self.sizes      = d['sizes']
        self._global_ids = d['global_ids']
        self._g2l = {int(gid): loc
                     for loc, gid in enumerate(self._global_ids)}
        self._dones_flat = [self.dones[t].view(-1)
                            for t in range(self.num_tasks)]
        self._cache_key = None


class ParallelRawReplayBuffer:
    """
    Per-task Raw Image Replay Buffer with (num_tasks, capacity_per_task, ...)
    structure, independent circular pointers, and task-balanced chunk sampling.
    """
    def __init__(self, num_tasks, capacity_per_task, img_height, img_width,
                 act_dim, device, global_task_ids):
        self.num_tasks = num_tasks
        self.capacity_per_task = capacity_per_task
        self.device = device

        self._global_ids = np.array(global_task_ids, dtype=np.int64)
        self._g2l = {}
        for local, gid in enumerate(self._global_ids):
            self._g2l[int(gid)] = local

        self.ptrs  = np.zeros(num_tasks, dtype=np.int64)
        self.sizes = np.zeros(num_tasks, dtype=np.int64)

        C = capacity_per_task
        self.imgs      = torch.zeros((num_tasks, C, 3, img_height, img_width),
                                     dtype=torch.uint8)
        self.next_imgs = torch.zeros((num_tasks, C, 3, img_height, img_width),
                                     dtype=torch.uint8)
        self.actions   = torch.zeros((num_tasks, C, act_dim), dtype=torch.float32)
        self.rewards   = torch.zeros((num_tasks, C, 1),       dtype=torch.float32)
        self.dones     = torch.zeros((num_tasks, C, 1),       dtype=torch.bool)

        self._dones_flat = [self.dones[t].view(-1) for t in range(num_tasks)]

        self._cache_key    = None
        self._gamma_vec    = None
        self._offsets       = None
        self._chunk_arange = None

    @property
    def total_size(self):
        return int(self.sizes.sum())

    @property
    def size(self):
        return self.total_size

    def add(self, img, action, reward, next_img, done, task_id):
        t = self._g2l[int(task_id)]
        p = int(self.ptrs[t])

        if isinstance(img, np.ndarray):    img    = torch.from_numpy(img)
        if isinstance(action, np.ndarray): action = torch.from_numpy(action)
        if isinstance(next_img, np.ndarray): next_img = torch.from_numpy(next_img)

        if img.dtype != torch.uint8:      img      = img.to(torch.uint8)
        if next_img.dtype != torch.uint8: next_img = next_img.to(torch.uint8)
        if len(img.shape) == 4 and img.shape[0] == 1:           img      = img.squeeze(0)
        if len(next_img.shape) == 4 and next_img.shape[0] == 1: next_img = next_img.squeeze(0)

        self.imgs[t, p]      = img
        self.next_imgs[t, p] = next_img
        self.actions[t, p]   = action.flatten()
        self.rewards[t, p]   = float(reward)
        self.dones[t, p]     = bool(done)

        self.ptrs[t]  = (p + 1) % self.capacity_per_task
        self.sizes[t] = min(self.sizes[t] + 1, self.capacity_per_task)

    def _ensure_cache(self, chunk_size, gamma):
        key = (chunk_size, gamma)
        if self._cache_key != key:
            self._cache_key    = key
            self._offsets       = torch.arange(chunk_size - 1)
            self._chunk_arange = torch.arange(chunk_size)
            self._gamma_vec    = (
                gamma ** torch.arange(chunk_size, dtype=torch.float32)
            ).view(1, chunk_size, 1)

    def _filter_task(self, cands, chunk_size, t):
        df = self._dones_flat[t]
        if chunk_size > 1:
            check = cands.unsqueeze(1) + self._offsets
            bad = df[check].any(dim=1)
        else:
            bad = torch.zeros(len(cands), dtype=torch.bool)

        if self.sizes[t] == self.capacity_per_task:
            ptr = int(self.ptrs[t])
            ptr_bad = (cands < ptr) & (ptr < cands + chunk_size)
            return cands[~bad & ~ptr_bad]
        return cands[~bad]

    def sample_chunk(self, batch_size, chunk_size, gamma):
        self._ensure_cache(chunk_size, gamma)

        sampleable = [t for t in range(self.num_tasks)
                      if self.sizes[t] >= chunk_size]
        ns = len(sampleable)
        if ns == 0:
            raise RuntimeError("No task has enough data for chunk sampling")

        per_task  = batch_size // ns
        remainder = batch_size % ns
        extra = set(np.random.choice(ns, remainder, replace=False).tolist()) \
                if remainder else set()

        parts_i, parts_ni, parts_a, parts_rew, parts_d, parts_tid = \
            [], [], [], [], [], []

        for i, t in enumerate(sampleable):
            n = per_task + (1 if i in extra else 0)
            if n == 0:
                continue
            mx = int(self.sizes[t]) - chunk_size
            if mx <= 0:
                continue

            cands = torch.from_numpy(
                np.random.randint(0, mx, size=n * 2, dtype=np.int64))
            idx = self._filter_task(cands, chunk_size, t)

            extra_n = n * 4
            while idx.shape[0] < n:
                ec = torch.from_numpy(
                    np.random.randint(0, mx, size=extra_n, dtype=np.int64))
                idx = torch.cat([idx, self._filter_task(ec, chunk_size, t)])
                extra_n *= 2
            idx = idx[:n]

            fin  = idx + chunk_size - 1
            grid = idx.unsqueeze(1) + self._chunk_arange

            parts_i.append(self.imgs[t][idx])
            parts_ni.append(self.next_imgs[t][fin])
            parts_a.append(self.actions[t][grid].reshape(n, -1))
            rseq = self.rewards[t][grid]
            parts_rew.append((rseq * self._gamma_vec).sum(dim=1))
            parts_d.append(self.dones[t][fin].float())
            parts_tid.append(
                torch.full((n,), self._global_ids[t], dtype=torch.long))

        imgs     = torch.cat(parts_i)
        nimgs    = torch.cat(parts_ni)
        acts     = torch.cat(parts_a)
        rews     = torch.cat(parts_rew)
        dones    = torch.cat(parts_d)
        tids     = torch.cat(parts_tid)

        perm = torch.randperm(tids.shape[0])
        return (
            imgs[perm].to(self.device, non_blocking=True),
            acts[perm].to(self.device, non_blocking=True),
            rews[perm].to(self.device, non_blocking=True),
            nimgs[perm].to(self.device, non_blocking=True),
            dones[perm].to(self.device, non_blocking=True),
            tids[perm].to(self.device, non_blocking=True),
        )

    def save(self, path):
        torch.save({
            'imgs': self.imgs, 'next_imgs': self.next_imgs,
            'actions': self.actions, 'rewards': self.rewards,
            'dones': self.dones,
            'ptrs': self.ptrs, 'sizes': self.sizes,
            'global_ids': self._global_ids,
        }, path)

    def load(self, path):
        d = torch.load(path)
        self.imgs      = d['imgs']
        self.next_imgs = d['next_imgs']
        self.actions   = d['actions']
        self.rewards   = d['rewards']
        self.dones     = d['dones']
        self.ptrs      = d['ptrs']
        self.sizes     = d['sizes']
        self._global_ids = d['global_ids']
        self._g2l = {int(gid): loc
                     for loc, gid in enumerate(self._global_ids)}
        self._dones_flat = [self.dones[t].view(-1)
                            for t in range(self.num_tasks)]
        self._cache_key = None


# =====================================================================
# Original flat Replay Buffers (kept for offline data / backward compat)
# =====================================================================

class FeatureReplayBuffer:
    """
    Feature-based Replay Buffer optimized for N-step (Chunk) sampling.
    Stores single transitions and assembles chunks efficiently using vectorized operations.
    """
    def __init__(self, capacity, repr_dim, act_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory (CPU Tensors for RAM efficiency)
        # repr_dim: Dimension of the representation (img_feat)
        self.reprs = torch.zeros((capacity, repr_dim), dtype=torch.float32)
        self.next_reprs = torch.zeros((capacity, repr_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.task_ids = torch.zeros((capacity, 1), dtype=torch.long)
        
        # Flat view of dones for efficient indexing (shared memory, no copy)
        self._dones_flat = self.dones.view(-1)
        
        # Cache for constant tensors reused across sample_chunk calls
        self._cache_key = None
        self._gamma_vec = None
        self._offsets = None
        self._chunk_arange = None

    def add(self, repr, action, reward, next_repr, done, task_id):
        """
        Add a single transition to the buffer.
        """
        if isinstance(repr, np.ndarray): repr = torch.from_numpy(repr)
        if isinstance(action, np.ndarray): action = torch.from_numpy(action)
        if isinstance(next_repr, np.ndarray): next_repr = torch.from_numpy(next_repr)
        
        self.reprs[self.ptr] = repr.flatten()
        self.next_reprs[self.ptr] = next_repr.flatten()
        self.actions[self.ptr] = action.flatten()
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = bool(done)
        self.task_ids[self.ptr] = int(task_id)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _ensure_cache(self, chunk_size, gamma):
        """Lazily create/update cached constant tensors for sample_chunk."""
        key = (chunk_size, gamma)
        if self._cache_key != key:
            self._cache_key = key
            self._offsets = torch.arange(chunk_size - 1)
            self._chunk_arange = torch.arange(chunk_size)
            self._gamma_vec = (
                gamma ** torch.arange(chunk_size, dtype=torch.float32)
            ).view(1, chunk_size, 1)

    def _filter_candidates(self, candidates, chunk_size):
        """Vectorized validity check: ptr safety + episode continuity."""
        # Episode continuity: no dones in [t, t+chunk_size-1)
        check_indices = candidates.unsqueeze(1) + self._offsets  # (N, K-1)
        has_early_done = self._dones_flat[check_indices].any(dim=1)  # (N,)
        
        # Ptr safety (only when buffer is full / circular)
        if self.size == self.capacity:
            ptr_unsafe = (candidates < self.ptr) & (self.ptr < candidates + chunk_size)
            valid_mask = ~has_early_done & ~ptr_unsafe
        else:
            valid_mask = ~has_early_done
        
        return candidates[valid_mask]

    def sample_chunk(self, batch_size, chunk_size, gamma):
        """
        Efficient Vectorized Chunk Sampling.
        1. Over-sample random indices (3x) for near-guaranteed single-pass.
        2. Compute validity masks in parallel (Overwrite safety, Done continuity).
        3. Gather data for valid indices with non-blocking GPU transfer.
        """
        self._ensure_cache(chunk_size, gamma)
        
        max_idx = self.size - chunk_size
        
        # numpy randint is faster than torch.randint on CPU
        candidates = torch.from_numpy(
            np.random.randint(0, max_idx, size=batch_size * 3, dtype=np.int64)
        )
        indices = self._filter_candidates(candidates, chunk_size)
        
        # Rare fallback: not enough valid indices
        if indices.shape[0] < batch_size:
            extra_n = batch_size * 8
            while indices.shape[0] < batch_size:
                extra_cands = torch.from_numpy(
                    np.random.randint(0, max_idx, size=extra_n, dtype=np.int64)
                )
                indices = torch.cat([indices, self._filter_candidates(extra_cands, chunk_size)])
                extra_n *= 2
        
        indices = indices[:batch_size]
        
        # --- Batch Data Gathering (Vectorized) ---
        final_indices = indices + chunk_size - 1
        gather_grid = indices.unsqueeze(1) + self._chunk_arange  # (B, K)
        
        reprs = self.reprs[indices]
        next_reprs = self.next_reprs[final_indices]
        action_chunk = self.actions[gather_grid].reshape(batch_size, -1)
        rewards_seq = self.rewards[gather_grid]
        discounted_reward = (rewards_seq * self._gamma_vec).sum(dim=1)
        chunk_dones = self.dones[final_indices].float()
        task_ids = self.task_ids[indices].squeeze(1)

        # Non-blocking GPU transfer (allows pipelining multiple transfers)
        return (
            reprs.to(self.device, non_blocking=True),
            action_chunk.to(self.device, non_blocking=True),
            discounted_reward.to(self.device, non_blocking=True),
            next_reprs.to(self.device, non_blocking=True),
            chunk_dones.to(self.device, non_blocking=True),
            task_ids.to(self.device, non_blocking=True)
        )

    def save(self, path):
        torch.save({
            'reprs': self.reprs,
            'next_reprs': self.next_reprs,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'task_ids': self.task_ids,
            'ptr': self.ptr,
            'size': self.size
        }, path)
        
    def load(self, path):
        data = torch.load(path)
        self.reprs = data['reprs']
        self.next_reprs = data['next_reprs']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        self.task_ids = data['task_ids']
        self.ptr = data['ptr']
        self.size = data['size']
        # Rebuild flat view and invalidate cache
        self._dones_flat = self.dones.view(-1)
        self._cache_key = None


class RawReplayBuffer:
    """
    Raw Image Replay Buffer optimized for N-step (Chunk) sampling.
    Stores single transitions with raw (3, H, W) uint8 images and assembles chunks efficiently using vectorized operations.
    """
    def __init__(self, capacity, img_height, img_width, act_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory (CPU Tensors for RAM efficiency)
        # Store raw images as (capacity, 3, H, W) uint8
        self.imgs = torch.zeros((capacity, 3, img_height, img_width), dtype=torch.uint8)
        self.next_imgs = torch.zeros((capacity, 3, img_height, img_width), dtype=torch.uint8)
        self.actions = torch.zeros((capacity, act_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.task_ids = torch.zeros((capacity, 1), dtype=torch.long)
        
        # Flat view of dones for efficient indexing (shared memory, no copy)
        self._dones_flat = self.dones.view(-1)
        
        # Cache for constant tensors reused across sample_chunk calls
        self._cache_key = None
        self._gamma_vec = None
        self._offsets = None
        self._chunk_arange = None

    def add(self, img, action, reward, next_img, done, task_id):
        """
        Add a single transition to the buffer.
        img and next_img should be (3, H, W) uint8 images.
        """
        if isinstance(img, np.ndarray): img = torch.from_numpy(img)
        if isinstance(action, np.ndarray): action = torch.from_numpy(action)
        if isinstance(next_img, np.ndarray): next_img = torch.from_numpy(next_img)
        
        # Ensure images are uint8 and correct shape (3, H, W)
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)
        if next_img.dtype != torch.uint8:
            next_img = next_img.to(torch.uint8)
        
        # Ensure shape is (3, H, W)
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if len(next_img.shape) == 4 and next_img.shape[0] == 1:
            next_img = next_img.squeeze(0)
        
        self.imgs[self.ptr] = img
        self.next_imgs[self.ptr] = next_img
        self.actions[self.ptr] = action.flatten()
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = bool(done)
        self.task_ids[self.ptr] = int(task_id)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _ensure_cache(self, chunk_size, gamma):
        """Lazily create/update cached constant tensors for sample_chunk."""
        key = (chunk_size, gamma)
        if self._cache_key != key:
            self._cache_key = key
            self._offsets = torch.arange(chunk_size - 1)
            self._chunk_arange = torch.arange(chunk_size)
            self._gamma_vec = (
                gamma ** torch.arange(chunk_size, dtype=torch.float32)
            ).view(1, chunk_size, 1)

    def _filter_candidates(self, candidates, chunk_size):
        """Vectorized validity check: ptr safety + episode continuity."""
        # Episode continuity: no dones in [t, t+chunk_size-1)
        check_indices = candidates.unsqueeze(1) + self._offsets  # (N, K-1)
        has_early_done = self._dones_flat[check_indices].any(dim=1)  # (N,)
        
        # Ptr safety (only when buffer is full / circular)
        if self.size == self.capacity:
            ptr_unsafe = (candidates < self.ptr) & (self.ptr < candidates + chunk_size)
            valid_mask = ~has_early_done & ~ptr_unsafe
        else:
            valid_mask = ~has_early_done
        
        return candidates[valid_mask]

    def sample_chunk(self, batch_size, chunk_size, gamma):
        """
        Efficient Vectorized Chunk Sampling.
        1. Over-sample random indices (2x) for near-guaranteed single-pass.
        2. Compute validity masks in parallel (Overwrite safety, Done continuity).
        3. Gather data for valid indices with non-blocking GPU transfer.
        """
        self._ensure_cache(chunk_size, gamma)
        
        max_idx = self.size - chunk_size
        
        # numpy randint is faster than torch.randint on CPU
        candidates = torch.from_numpy(
            np.random.randint(0, max_idx, size=batch_size * 2, dtype=np.int64)
        )
        indices = self._filter_candidates(candidates, chunk_size)
        
        # Rare fallback: not enough valid indices
        if indices.shape[0] < batch_size:
            extra_n = batch_size * 2
            while indices.shape[0] < batch_size:
                extra_cands = torch.from_numpy(
                    np.random.randint(0, max_idx, size=extra_n, dtype=np.int64)
                )
                indices = torch.cat([indices, self._filter_candidates(extra_cands, chunk_size)])
                extra_n *= 2
        
        indices = indices[:batch_size]
        
        # --- Batch Data Gathering (Vectorized) ---
        final_indices = indices + chunk_size - 1
        gather_grid = indices.unsqueeze(1) + self._chunk_arange  # (B, K)
        
        imgs = self.imgs[indices]
        next_imgs = self.next_imgs[final_indices]
        action_chunk = self.actions[gather_grid].reshape(batch_size, -1)
        rewards_seq = self.rewards[gather_grid]
        discounted_reward = (rewards_seq * self._gamma_vec).sum(dim=1)
        chunk_dones = self.dones[final_indices].float()
        task_ids = self.task_ids[indices].squeeze(1)

        # Non-blocking GPU transfer (allows pipelining multiple transfers)
        return (
            imgs.to(self.device, non_blocking=True),
            action_chunk.to(self.device, non_blocking=True),
            discounted_reward.to(self.device, non_blocking=True),
            next_imgs.to(self.device, non_blocking=True),
            chunk_dones.to(self.device, non_blocking=True),
            task_ids.to(self.device, non_blocking=True)
        )

    def save(self, path):
        torch.save({
            'imgs': self.imgs,
            'next_imgs': self.next_imgs,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'task_ids': self.task_ids,
            'ptr': self.ptr,
            'size': self.size
        }, path)
        
    def load(self, path):
        data = torch.load(path)
        self.imgs = data['imgs']
        self.next_imgs = data['next_imgs']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        self.task_ids = data['task_ids']
        self.ptr = data['ptr']
        self.size = data['size']
        # Rebuild flat view and invalidate cache
        self._dones_flat = self.dones.view(-1)
        self._cache_key = None


# =====================================================================
# BRC-style Parallel Replay Buffer (shared insert index, numpy-based)
# =====================================================================

class BRCReplayBuffer:
    """
    Replay buffer faithful to BRC's ParallelReplayBuffer.

    Layout: (num_tasks, capacity, ...) with a single shared insert_index.
    All tasks insert at the same index each step (parallel lockstep).
    Sampling returns (num_batches, batch_size, ...) numpy arrays.
    """

    def __init__(self, num_tasks, capacity, state_dim, act_dim):
        self.num_tasks = num_tasks
        self.capacity = capacity
        self.observations      = np.zeros((num_tasks, capacity, state_dim), dtype=np.float64)
        self.actions            = np.zeros((num_tasks, capacity, act_dim),   dtype=np.float32)
        self.rewards            = np.zeros((num_tasks, capacity),            dtype=np.float32)
        self.masks              = np.zeros((num_tasks, capacity),            dtype=np.float32)
        self.next_observations = np.zeros((num_tasks, capacity, state_dim), dtype=np.float64)
        self.size = 0
        self.insert_index = 0

    def insert(self, observations, actions, rewards, masks, next_observations):
        """Insert one parallel transition (all tasks at the same index).

        Args:
            observations:      (num_tasks, state_dim)
            actions:           (num_tasks, act_dim)
            rewards:           (num_tasks,)
            masks:             (num_tasks,)
            next_observations: (num_tasks, state_dim)
        """
        idx = self.insert_index
        self.observations[:, idx]      = observations
        self.actions[:, idx]           = actions
        self.rewards[:, idx]           = rewards
        self.masks[:, idx]             = masks
        self.next_observations[:, idx] = next_observations
        self.insert_index = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, num_batches):
        """Sample random batches across all tasks.

        Returns numpy arrays shaped (num_batches, batch_size, ...).
        task_ids are the local task indices (0..num_tasks-1).
        """
        indx = np.random.randint(self.size * self.num_tasks,
                                 size=(num_batches, batch_size))
        task_indx, sample_indx = np.divmod(indx, self.size)
        return {
            'observations':      self.observations[task_indx, sample_indx],
            'actions':           self.actions[task_indx, sample_indx],
            'rewards':           self.rewards[task_indx, sample_indx],
            'masks':             self.masks[task_indx, sample_indx],
            'next_observations': self.next_observations[task_indx, sample_indx],
            'task_ids':          task_indx,
        }
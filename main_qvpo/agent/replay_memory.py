import numpy as np
import torch


class ReplayMemory():
    """Buffer to store environment transitions."""
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        self.states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.actions = np.empty((self.capacity, int(action_dim)), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.masks = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    @property
    def size(self):
        return self.capacity if self.full else self.idx

    def append(self, state, action, reward, next_state, mask):

        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_states[self.idx], next_state)
        np.copyto(self.masks[self.idx], mask)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        
        # Add this dynamic priority expansion logic right after saving tensors:
        if hasattr(self, '_priorities'):
            # Check if the buffer has expanded beyond your current priorities list size
            current_buffer_size = self.size
            if len(self._priorities) < current_buffer_size:
                # Pad the priorities list with max priority (1.0) to match the new size
                padding_size = current_buffer_size - len(self._priorities)
                self._priorities = np.append(self._priorities, np.ones(padding_size, dtype=np.float32))

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        states = torch.as_tensor(self.states[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device)
        masks = torch.as_tensor(self.masks[idxs], device=self.device)

        return states, actions, rewards, next_states, masks

    def sample_by_idx(self, idxs):
        """Sample specific rows by index array."""
        states = torch.as_tensor(self.states[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device)
        masks = torch.as_tensor(self.masks[idxs], device=self.device)

        return states, actions, rewards, next_states, masks


class DiffusionMemory():
    """Buffer to store best actions."""
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        self.states = np.empty((self.capacity, int(state_dim)), dtype=np.float32)
        self.best_actions = np.empty((self.capacity, int(action_dim)), dtype=np.float32)

        self.idx = 0
        self.full = False

    @property
    def size(self):
        return self.capacity if self.full else self.idx

    def append(self, state, action):

        np.copyto(self.states[self.idx], state)
        np.copyto(self.best_actions[self.idx], action)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        states = torch.as_tensor(self.states[idxs], device=self.device)
        best_actions = torch.as_tensor(self.best_actions[idxs], device=self.device)

        best_actions.requires_grad_(True)

        return states, best_actions, idxs

    def replace(self, idxs, best_actions):
        np.copyto(self.best_actions[idxs], best_actions)

class OfflineBuffer:
    """
    Static offline dataset — stays on CPU with pinned memory.

    For large MuJoCo datasets (up to 1M × 376 for Humanoid) keeping the
    buffer on GPU wastes VRAM.  Batches are moved to device inside sample()
    using non_blocking=True so the copy overlaps with GPU compute.

    Expects an .npz file with keys:
        states, actions, rewards, next_states, dones
    All shapes: (N, dim) for states/actions/next_states, (N,) or (N,1) for
    rewards and dones — both are normalised to (N, 1) at load time.
    """

    def __init__(self, path: str, device: torch.device):
        raw = np.load(path)

        self.device = device

        # Load everything to CPU, pin for fast async GPU transfer
        def _t(key, dtype=torch.float32):
            return torch.tensor(raw[key], dtype=dtype).pin_memory()

        self.states      = _t("states")
        self.actions     = _t("actions")
        self.next_states = _t("next_states")

        # Normalise rewards / dones to (N, 1)
        r = torch.tensor(raw["rewards"], dtype=torch.float32)
        d = torch.tensor(raw["dones"],   dtype=torch.float32)
        self.rewards = r.view(-1, 1).pin_memory()
        self.dones   = d.view(-1, 1).pin_memory()

        self.size = len(self.states)
        print(f"[OfflineBuffer] {self.size:,} transitions | "
              f"state={tuple(self.states.shape[1:])}  "
              f"action={tuple(self.actions.shape[1:])}")

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,))
        # non_blocking so GPU can overlap compute with this transfer
        to = lambda t: t[idx].to(self.device, non_blocking=True)
        return {
            "states":      to(self.states),
            "actions":     to(self.actions),
            "rewards":     to(self.rewards),
            "next_states": to(self.next_states),
            "dones":       to(self.dones),
        }

def _offline_sample_by_idx(buf: OfflineBuffer, idx: np.ndarray) -> dict:
    """Sample specific rows from the CPU offline buffer by numpy index array."""
    t = torch.from_numpy(idx).long()
    to = lambda x: x[t].to(buf.device, non_blocking=True)
    return {
        "states":      to(buf.states),
        "actions":     to(buf.actions),
        "rewards":     to(buf.rewards),
        "next_states": to(buf.next_states),
        "dones":       to(buf.dones),
    }

# Attach as method at module load
OfflineBuffer.sample_by_idx = _offline_sample_by_idx

class HyQMixer:
    """
    Hy-Q priority-weighted offline/online batch mixer.

    Fix: replace=True in np.random.choice.
      replace=False requires a full argsort over the priority array: O(N log N).
      replace=True uses NumPy's Vose alias method: O(N) construction, O(1) draw.
      For a 1M offline buffer this is the difference between ~30ms and ~0.1ms
      per mixer.sample() call.
    """

    def __init__(
        self,
        offline_buf, #: OfflineBuffer,
        online_buf,
        beta_start:   float = 1.0,
        beta_end:     float = 0.25,
        anneal_steps: int   = 50_000,
        td_alpha:     float = 0.6,
    ):
        self.offline      = online_buf
        self.online       = online_buf
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.anneal_steps = anneal_steps
        self.td_alpha     = td_alpha
        self._step        = 0
        self._priorities  = np.ones(offline_buf.size, dtype=np.float32) if offline_buf is not None else np.ones(1, dtype=np.float32)
        self.device       = None

    @property
    def beta(self) -> float:
        frac = min(self._step / max(self.anneal_steps, 1), 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        flattened_errors = np.abs(td_errors.flatten())
        self._priorities[indices] = (flattened_errors + 1e-6) ** self.td_alpha

    def sample(self, batch_size: int):
        """
        Sample a batch mixing offline and online data based on priority weights.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, masks)
        """
        self._step += 1
        
        # 1. Handle Online-Only Sampling
        if self.offline is None:
            return self.online.sample(batch_size)
        
        n_offline = int(round(self.beta * batch_size))
        n_online  = batch_size - n_offline
        
        # 2. Sample from Offline Buffer
        if n_offline > 0:
            probs = self._priorities / self._priorities.sum()
            # Draw indices based on priorities
            idx = np.random.choice(self.offline.size, size=n_offline, replace=True, p=probs)
            self._offline_idx = idx  # Keep track for updates
            
            off_data = _offline_sample_by_idx(self.offline, idx)
        
        # 3. Sample from Online Buffer
        if n_online > 0:
            # Assumes online buffer returns a tuple: (s, a, r, ns, m)
            on_s, on_a, on_r, on_ns, on_m = self.online.sample(n_online)

        # 4. Merge Data or Return pure subsets
        if n_online == 0:
            # Pure offline data tuple unpack from dictionary
            return (
                off_data["states"], 
                off_data["actions"], 
                off_data["rewards"], 
                off_data["next_states"], 
                off_data["dones"] # masks map to dones in OfflineBuffer
            )
        elif n_offline == 0:
            return on_s, on_a, on_r, on_ns, on_m
        else:
            # Combine tensors from both buffers
            states = torch.cat([off_data["states"], on_s], dim=0)
            actions = torch.cat([off_data["actions"], on_a], dim=0)
            rewards = torch.cat([off_data["rewards"], on_r], dim=0)
            next_states = torch.cat([off_data["next_states"], on_ns], dim=0)
            masks = torch.cat([off_data["dones"], on_m], dim=0)
            return states, actions, rewards, next_states, masks


    def update(self, states, actions, rewards, next_states, masks, critic, actor, actor_target, device):
        """
        Update priorities based on TD errors computed from critic.
        
        Args:
            states, actions, rewards, next_states, masks: batch tensors from sample()
            critic: current critic network
            actor: current actor network
            actor_target: target actor network
            device: device to run on
        """
        if self.offline is None or self._offline_idx is None:
            return
        
        # Compute TD error for offline samples
        with torch.no_grad():
            # Get target Q-values
            next_actions = actor_target(next_states, eval=False, q_func=critic)
            target_q1, target_q2 = critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + masks * target_q
            
            # Get current Q-values for offline samples
            current_q1, current_q2 = critic(states[:len(self._offline_idx)], 
                                            actions[:len(self._offline_idx)])
            
            # Compute TD errors
            td_errors = torch.abs(target_q[:len(self._offline_idx)] - 
                                 torch.min(current_q1, current_q2)).cpu().numpy()
        
        # Update priorities
        self.update_priorities(self._offline_idx, td_errors)

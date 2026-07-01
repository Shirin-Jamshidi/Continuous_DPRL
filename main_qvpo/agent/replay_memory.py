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
        offline_buf,
        online_buf,
        use_offline:  bool = False,
        beta_start:   float = 1.0,
        beta_end:     float = 0.25,
        anneal_steps: int   = 50_000,
        td_alpha:     float = 0.6,
    ):
        self.offline      = online_buf
        self.online       = online_buf
        self.use_offline  = use_offline
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
        self._priorities[indices] = (np.abs(td_errors) + 1e-6) ** self.td_alpha

    def sample(self, batch_size: int):
        """
        Sample a batch mixing offline and online data based on priority weights.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, masks)
        """
        self._step += 1
        
        if not self.use_offline or self.offline is None:
            # Only use online buffer
            return self.online.sample(batch_size)
        
        n_offline = int(round(self.beta * batch_size))
        n_online  = batch_size - n_offline
        batches = []
        self._offline_idx = None

        if n_offline > 0:
            probs = self._priorities / self._priorities.sum()
            # replace=True: O(N) alias method vs O(N log N) for replace=False
            self._offline_idx = np.random.choice(
                self.offline.size, size=n_offline, replace=True, p=probs
            )
            batches.append(self.offline.sample_by_idx(self._offline_idx))

        if n_online > 0:
            src = self.online if self.online.size >= n_online else self.offline
            batches.append(src.sample(n_online))

        # Concatenate all tensors along batch dimension
        if len(batches) == 1:
            return batches[0]
        
        states = torch.cat([b[0] for b in batches], dim=0)
        actions = torch.cat([b[1] for b in batches], dim=0)
        rewards = torch.cat([b[2] for b in batches], dim=0)
        next_states = torch.cat([b[3] for b in batches], dim=0)
        masks = torch.cat([b[4] for b in batches], dim=0)
        
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
        if not self.use_offline or self.offline is None or self._offline_idx is None:
            return
        
        # Compute TD error for offline samples
        with torch.no_grad():
            # Get target Q-values
            next_actions = actor_target(next_states, eval=False)
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
    
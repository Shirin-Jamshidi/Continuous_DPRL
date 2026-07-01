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


class PriorityMixer:
    """
    TD-error prioritized sampling wrapper around a single (online) ReplayMemory.

    The offline buffer / online-offline mixing has been removed entirely.
    This class now does one thing: it draws a batch from `online_buf` using
    proportional TD-error priorities (Prioritized Experience Replay, Schaul et
    al. 2016), then lets you feed the resulting TD errors back in to update
    those priorities.

    Sampling still uses np.random.choice(..., replace=True), which is the
    O(N) Vose alias method rather than an O(N log N) argsort-based approach,
    so this remains cheap even for large buffers.

    Usage is unchanged from the training loop's perspective:

        states, actions, rewards, next_states, masks = self.mixer.sample(batch_size)
        ...
        self.mixer.update(states, actions, rewards, next_states, masks,
                           self.critic, self.actor, self.actor_target, self.device)
    """

    def __init__(
        self,
        online_buf,
        td_alpha: float = 0.6,
        eps: float = 1e-6,
    ):
        self.online = online_buf
        self.capacity = online_buf.capacity
        self.device = online_buf.device
        self.td_alpha = td_alpha
        self.eps = eps

        # Per-slot priorities, aligned with the underlying ReplayMemory's
        # circular buffer indices.
        self._priorities = np.zeros(self.capacity, dtype=np.float32)
        self._max_priority = 1.0  # newly written transitions start "max priority"

        # Bookkeeping to detect which slots were newly written since the
        # last sample() call, since ReplayMemory.append() doesn't know about
        # priorities.
        self._last_idx = 0
        self._last_full = False

        self._sampled_idx = None

    def _sync_new_entries(self):
        """Assign max priority to any buffer slots written since last sync."""
        cur_idx = self.online.idx
        cur_full = self.online.full

        if cur_full and not self._last_full:
            # Buffer wrapped for the first time: everything from the old
            # write pointer to the end of the array is new.
            self._priorities[self._last_idx:] = self._max_priority
            self._last_idx = 0
            self._last_full = True

        if cur_idx >= self._last_idx:
            self._priorities[self._last_idx:cur_idx] = self._max_priority
        else:
            # Write pointer wrapped around during this interval.
            self._priorities[self._last_idx:] = self._max_priority
            self._priorities[:cur_idx] = self._max_priority

        self._last_idx = cur_idx

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.eps) ** self.td_alpha
        self._priorities[indices] = priorities
        self._max_priority = max(self._max_priority, float(priorities.max()))

    def sample(self, batch_size: int):
        """
        Sample a batch from the online buffer using TD-error priorities.

        Returns:
            Tuple of (states, actions, rewards, next_states, masks)
        """
        self._sync_new_entries()

        size = self.online.size
        probs = self._priorities[:size]
        probs = probs / probs.sum()

        idxs = np.random.choice(size, size=batch_size, replace=True, p=probs)
        self._sampled_idx = idxs

        return self.online.sample_by_idx(idxs)

    def update(self, states, actions, rewards, next_states, masks, critic, actor, actor_target, device):
        """
        Compute TD errors for the last-sampled batch and update priorities.

        Args:
            states, actions, rewards, next_states, masks: batch tensors from sample()
            critic: current critic network
            actor: current actor network (unused, kept for signature parity)
            actor_target: target actor network
            device: device to run on (unused, kept for signature parity)
        """
        if self._sampled_idx is None:
            return

        with torch.no_grad():
            next_actions = actor_target(next_states, eval=False)
            target_q1, target_q2 = critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + masks * target_q

            current_q1, current_q2 = critic(states, actions)

            td_errors = torch.abs(target_q - torch.min(current_q1, current_q2))
            td_errors = td_errors.squeeze(-1).cpu().numpy()

        self.update_priorities(self._sampled_idx, td_errors)
        self._sampled_idx = None
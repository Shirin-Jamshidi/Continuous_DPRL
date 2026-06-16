"""
Diffusion Q-Learning for Continuous CartPole
=============================================
Combines:
  - QVPO  : Q-weighted VLO loss with Gaussian (continuous) diffusion policy
  - Hy-Q  : Hybrid offline + online Q-learning with importance-weighted mixing

Key change vs. previous version
---------------------------------
  The previous discrete CartPole + sign-mapping approach collapsed to ~20-30
  return because the continuous Gaussian diffusion policy and the discrete
  action space are fundamentally mismatched: a tanh-squashed scalar loses all
  gradient signal when discretised by a step function.

  Following the professor's suggestion, we instead **replace the CartPole
  dynamics model** with a native continuous-force variant:

    ContinuousCartPoleEnv
      - Action space : Box([-force_mag], [+force_mag], dtype=float32)
      - Physics      : identical equations of motion as CartPole-v1, but the
                       scalar force f ∈ [-force_mag, +force_mag] is applied
                       directly to the cart instead of the bang-bang ±10 N.
      - Reward       : +1 for every step the pole stays up (same as v1)
      - Termination  : same thresholds as v1 (|pole|>12°, |cart|>2.4 m)
      - Max steps    : 500 (same as v1)

  The offline demo data (discrete 0/1) is remapped at load time:
      discrete 0 → force  -force_mag   (push left)
      discrete 1 → force  +force_mag   (push right)
  These are valid points in the continuous action space so there is no
  information loss and the diffusion model can learn to interpolate between
  them as training progresses.

Architecture
------------
  ContinuousCartPoleEnv  : native continuous-force CartPole
  OfflineBuffer          : loads demo data, converts actions to continuous
  OnlineBuffer           : ring buffer for online experience
  GaussianDiffusion      : DDPM with linear β schedule, ε-prediction network
  EpsilonNet             : MLP(a_t, s, t) → ε̂  (noise prediction)
  TwinQNetwork           : Q(s,a) → scalar, twin critics (SAC-style)
  HyQMixer               : priority-weighted offline/online batch mixing
  DiffusionQLTrainer     : offline pretraining → online finetuning loop
"""

import os
import copy
import math
import random
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Continuous CartPole Environment
#     Replaces gymnasium's Discrete(2) action space with Box(1,) so the
#     Gaussian diffusion policy operates in its natural continuous domain.
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousCartPoleEnv:
    """
    CartPole with a continuous scalar force action.

    Dynamics
    --------
    Identical to CartPole-v1 (Barto, Sutton & Anderson 1983) with the single
    change that the applied force is the raw network output f ∈ [-F, +F]
    rather than a bang-bang ±F signal.

    State  : [cart_pos, cart_vel, pole_angle, pole_ang_vel]   (same as v1)
    Action : scalar force f ∈ [-force_mag, +force_mag]
    Reward : +1.0 every step the pole stays up
    Done   : |pole| > 12° OR |cart| > 2.4 m OR steps ≥ max_steps
    """

    # Physics constants (same as CartPole-v1)
    GRAVITY    = 9.8
    MASSCART   = 1.0
    MASSPOLE   = 0.1
    TOTAL_MASS = MASSCART + MASSPOLE
    HALF_LEN   = 0.5          # half the pole length
    POLEMASS_LEN = MASSPOLE * HALF_LEN
    TAU        = 0.02         # seconds per step

    # Termination thresholds (same as CartPole-v1)
    THETA_THRESHOLD = 12 * 2 * math.pi / 360   # 12 degrees in radians
    X_THRESHOLD     = 2.4                       # metres

    def __init__(self, force_mag: float = 10.0, max_steps: int = 500, seed: int = 42):
        self.force_mag = force_mag
        self.max_steps = max_steps
        self._rng      = np.random.RandomState(seed)
        self.state     = None
        self._step_count = 0

        # Spaces (for external compatibility checks)
        self.observation_space_shape = (4,)
        self.action_space_low  = np.array([-force_mag], dtype=np.float32)
        self.action_space_high = np.array([ force_mag], dtype=np.float32)

    # ------------------------------------------------------------------
    def seed(self, s: int):
        self._rng = np.random.RandomState(s)

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        # Uniform in [-0.05, 0.05] for all state variables (same as v1)
        self.state       = self._rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        self._step_count = 0
        return self.state.copy(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        action : np.ndarray of shape (1,) or scalar, in [-force_mag, +force_mag]
        """
        f = float(np.clip(action, -self.force_mag, self.force_mag))

        x, x_dot, theta, theta_dot = self.state

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Standard CartPole equations of motion
        tmp        = (f + self.POLEMASS_LEN * theta_dot ** 2 * sin_t) / self.TOTAL_MASS
        theta_acc  = (self.GRAVITY * sin_t - cos_t * tmp) / (
            self.HALF_LEN * (4.0 / 3.0 - self.MASSPOLE * cos_t ** 2 / self.TOTAL_MASS)
        )
        x_acc      = tmp - self.POLEMASS_LEN * theta_acc * cos_t / self.TOTAL_MASS

        # Euler integration
        x         += self.TAU * x_dot
        x_dot     += self.TAU * x_acc
        theta     += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._step_count += 1

        terminated = bool(
            abs(x)     > self.X_THRESHOLD or
            abs(theta) > self.THETA_THRESHOLD
        )
        truncated = (self._step_count >= self.max_steps)

        reward = 1.0 if not terminated else 0.0
        return self.state.copy(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Replay Buffers
# ══════════════════════════════════════════════════════════════════════════════

class OfflineBuffer:
    """
    Loads the static demonstration dataset and converts discrete actions
    {0, 1} → continuous forces {-force_mag, +force_mag}.

    Rationale: the demo data came from a discrete Q-learning agent, but our
    continuous CartPole environment accepts scalar forces.  Mapping
        0 → -force_mag  (push left at full force)
        1 → +force_mag  (push right at full force)
    preserves the original policy's intent exactly and places the demo
    actions at valid extremes of the continuous action space.
    """

    def __init__(self, path: str, device: torch.device, force_mag: float = 10.0):
        raw = np.load(path)

        states     = torch.tensor(raw["states"],      dtype=torch.float32)
        # actions_d  = torch.tensor(raw["actions"],     dtype=torch.float32)  # 0 or 1
        rewards    = torch.tensor(raw["rewards"],     dtype=torch.float32)
        next_states= torch.tensor(raw["next_states"], dtype=torch.float32)

        # # Discrete {0,1} → continuous {-force_mag, +force_mag}, shape (N,1)
        # actions_c = (actions_d * 2.0 - 1.0) * force_mag   # (N,)
        # actions_c = actions_c.unsqueeze(-1)                 # (N, 1)

        actions = torch.tensor(raw["actions"], dtype=torch.float32)

        # ✅ ensure shape is (N,1)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        elif actions.dim() == 3:
            actions = actions.squeeze(-1)


        # No 'dones' key — all transitions are mid-episode (reward = 1 always)
        dones = torch.zeros(len(states), dtype=torch.float32)

        self.states      = states.to(device)
        self.actions     = actions.to(device)
        self.rewards     = rewards.to(device)
        self.next_states = next_states.to(device)
        self.dones       = dones.to(device)
        self.size        = len(states)
        self.device      = device

        print(f"[OfflineBuffer] {self.size:,} transitions | "
              f"actions remapped: 0→{-force_mag:.1f} N, 1→{+force_mag:.1f} N")

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "states":      self.states[idx],
            "actions":     self.actions[idx],      # (B, 1) continuous
            "rewards":     self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones":       self.dones[idx],
        }


class OnlineBuffer:
    """GPU ring-buffer for online experience from ContinuousCartPoleEnv."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 device: torch.device):
        self.capacity = capacity
        self.device   = device
        self._ptr     = 0
        self._full    = False

        self.states      = torch.zeros((capacity, state_dim),  device=device)
        self.actions     = torch.zeros((capacity, action_dim), device=device)
        self.rewards     = torch.zeros((capacity,),            device=device)
        self.next_states = torch.zeros((capacity, state_dim),  device=device)
        self.dones       = torch.zeros((capacity,),            device=device)

    @property
    def size(self) -> int:
        return self.capacity if self._full else self._ptr

    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, next_state: np.ndarray, done: float):
        i = self._ptr
        self.states[i]      = torch.as_tensor(state,      dtype=torch.float32, device=self.device)
        self.actions[i]     = torch.as_tensor(action,     dtype=torch.float32, device=self.device)
        self.rewards[i]     = reward
        self.next_states[i] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[i]       = done
        self._ptr = (self._ptr + 1) % self.capacity
        if self._ptr == 0:
            self._full = True

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "states":      self.states[idx],
            "actions":     self.actions[idx],
            "rewards":     self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones":       self.dones[idx],
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Hy-Q Mixer
# ══════════════════════════════════════════════════════════════════════════════

class HyQMixer:
    """
    Hy-Q hybrid batch sampler.

    β decays linearly from β_start → β_end over anneal_steps gradient steps.
    Offline samples are weighted by |TD-error|^α (priority weighting).
    """

    def __init__(
        self,
        offline_buf:  OfflineBuffer,
        online_buf:   OnlineBuffer,
        beta_start:   float = 1.0,
        beta_end:     float = 0.25,
        anneal_steps: int   = 50_000,
        td_alpha:     float = 0.6,
    ):
        self.offline      = offline_buf
        self.online       = online_buf
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.anneal_steps = anneal_steps
        self.td_alpha     = td_alpha
        self._step        = 0
        self._priorities  = np.ones(offline_buf.size, dtype=np.float32)

    @property
    def beta(self) -> float:
        frac = min(self._step / max(self.anneal_steps, 1), 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        self._priorities[indices] = (np.abs(td_errors) + 1e-6) ** self.td_alpha

    def sample(self, batch_size: int) -> Tuple[dict, Optional[np.ndarray]]:
        self._step += 1
        n_offline = int(round(self.beta * batch_size))
        n_online  = batch_size - n_offline
        batches, offline_idx = [], None

        if n_offline > 0:
            probs = self._priorities / self._priorities.sum()
            offline_idx = np.random.choice(
                self.offline.size, size=n_offline, replace=False, p=probs
            )
            idx_t = torch.tensor(offline_idx, device=self.offline.device)
            batches.append({
                "states":      self.offline.states[idx_t],
                "actions":     self.offline.actions[idx_t],
                "rewards":     self.offline.rewards[idx_t],
                "next_states": self.offline.next_states[idx_t],
                "dones":       self.offline.dones[idx_t],
            })

        if n_online > 0:
            src = self.online if self.online.size >= n_online else self.offline
            batches.append(src.sample(n_online))

        merged = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}
        return merged, offline_idx


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Gaussian Diffusion (continuous action space)
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) *
            torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class EpsilonNet(nn.Module):
    """
    ε_θ(ã_t, s, t) → ε̂  — noise prediction network.

    Input  : cat(ã_t, s, t_emb)   noisy action + state + time embedding
    Output : ε̂ ∈ R^{action_dim}  same shape as the action
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        hidden_dim:   int = 256,
        time_emb_dim: int = 16,
        n_steps:      int = 5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_steps    = n_steps
        self.time_emb   = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = action_dim + state_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = list(self.net.children())[-1]
        nn.init.uniform_(last.weight, -1e-3, 1e-3)
        nn.init.zeros_(last.bias)

    def forward(self, a_t: torch.Tensor, state: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        x = torch.cat([a_t, state, t_emb], dim=-1)
        return self.net(x)


class GaussianDiffusion:
    """
    DDPM with linear β schedule for continuous actions.

    Schedule : T steps, β linearly spaced in [beta_min, beta_max].
    Forward  : q(ã_t | a_0) = N(√ᾱ_t · a_0,  (1−ᾱ_t)·I)
    Reverse  : p_θ(ã_{t-1} | ã_t, s) via EpsilonNet

    The action is NOT tanh-squashed here — the force magnitude is bounded
    by environment clipping (±force_mag), which is a more honest inductive
    bias for physics-based control than tanh.
    """

    def __init__(self, n_steps: int = 5, beta_min: float = 0.1,
                 beta_max: float = 0.5):
        self.T    = n_steps
        betas     = torch.linspace(beta_min, beta_max, n_steps)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])

        self.betas          = betas
        self.alpha_bar      = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_ab        = alpha_bar.sqrt()
        self.sqrt_1mab      = (1.0 - alpha_bar).sqrt()
        self.posterior_var  = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * betas
        self._device        = None

    def to(self, device: torch.device) -> "GaussianDiffusion":
        for attr in ["betas", "alpha_bar", "alpha_bar_prev",
                     "sqrt_ab", "sqrt_1mab", "posterior_var"]:
            setattr(self, attr, getattr(self, attr).to(device))
        self._device = device
        return self

    # ── Forward process ────────────────────────────────────────────────────

    def q_sample(self, a0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (ã_t, ε) — noisy action and the noise added."""
        if noise is None:
            noise = torch.randn_like(a0)
        s_ab   = self.sqrt_ab[t - 1].view(-1, 1)
        s_1mab = self.sqrt_1mab[t - 1].view(-1, 1)
        return s_ab * a0 + s_1mab * noise, noise

    # ── Reverse step ───────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample_step(self, eps_net: EpsilonNet, a_t: torch.Tensor,
                      state: torch.Tensor, t_val: int) -> torch.Tensor:
        B  = a_t.shape[0]
        t  = torch.full((B,), t_val, dtype=torch.long, device=a_t.device)
        eps_hat = eps_net(a_t, state, t)

        s_ab   = self.sqrt_ab[t_val - 1]
        s_1mab = self.sqrt_1mab[t_val - 1]
        a0_hat = (a_t - s_1mab * eps_hat) / s_ab

        ab      = self.alpha_bar[t_val - 1]
        ab_prev = self.alpha_bar_prev[t_val - 1]
        beta_t  = self.betas[t_val - 1]
        mu = (ab_prev.sqrt() * beta_t * a0_hat +
              (1.0 - ab_prev) * (1.0 - beta_t).sqrt() * a_t) / (1.0 - ab)

        if t_val == 1:
            return mu
        sigma = self.posterior_var[t_val - 1].sqrt()
        return mu + sigma * torch.randn_like(a_t)

    # ── Full reverse chain: action sampling ────────────────────────────────

    @torch.no_grad()
    def p_sample(self, eps_net: EpsilonNet, state: torch.Tensor,
                 n_samples: int = 1) -> torch.Tensor:
        """
        Sample n_samples actions per state.
        Returns (B * n_samples, action_dim).
        """
        if n_samples > 1:
            state = state.repeat_interleave(n_samples, dim=0)
        a_t = torch.randn(state.shape[0], eps_net.action_dim, device=state.device)
        for t_val in reversed(range(1, self.T + 1)):
            a_t = self.p_sample_step(eps_net, a_t, state, t_val)
        return a_t

    # ── Q-weighted VLO loss (QVPO Eq. 6) ──────────────────────────────────

    def q_weighted_vlo_loss(self, eps_net: EpsilonNet, a_sel: torch.Tensor,
                            state: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
        """
        L(θ) = E_{t,ε}[ ω_eq(s,a) · ||ε − ε_θ(√ᾱ_t·a + √(1−ᾱ_t)·ε, s, t)||² ]
        """
        B  = a_sel.shape[0]
        t  = torch.randint(1, self.T + 1, (B,), device=a_sel.device)
        a_t, noise = self.q_sample(a_sel, t)
        eps_hat    = eps_net(a_t, state, t)
        per_sample = ((noise - eps_hat) ** 2).mean(dim=-1)
        return (weights * per_sample).mean()

    # ── Entropy regularisation (QVPO Eq. 10) ──────────────────────────────

    def entropy_loss(self, eps_net: EpsilonNet, state: torch.Tensor,
                     omega_ent_s: torch.Tensor,
                     n_uniform: int, force_mag: float) -> torch.Tensor:
        """
        Push policy toward higher entropy by fitting uniform actions.
        Uniform actions drawn from U(-force_mag, +force_mag) — the full
        physical range of the continuous CartPole action space.
        """
        B = state.shape[0]
        state_rep    = state.repeat_interleave(n_uniform, dim=0)
        omega_rep    = omega_ent_s.repeat_interleave(n_uniform, dim=0)
        a_unif = (torch.rand(B * n_uniform, eps_net.action_dim,
                             device=state.device) * 2.0 - 1.0) * force_mag
        t = torch.randint(1, self.T + 1, (B * n_uniform,), device=state.device)
        a_t, noise = self.q_sample(a_unif, t)
        eps_hat    = eps_net(a_t, state_rep, t)
        per_sample = ((noise - eps_hat) ** 2).mean(dim=-1)
        return (omega_rep * per_sample).mean()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Twin Q-Network  Q(s, a) → scalar
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """Single Q-network for a continuous (state, action) pair."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),             nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = list(self.net.children())[-1]
        nn.init.uniform_(last.weight, -3e-3, 3e-3)
        nn.init.zeros_(last.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))  # (B, 1)


def q_min(q1: QNetwork, q2: QNetwork,
          state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """min(Q1, Q2) as (B,) — pessimistic estimate."""
    return torch.min(q1(state, action), q2(state, action)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main Trainer
# ══════════════════════════════════════════════════════════════════════════════

class DiffusionQLTrainer:
    """
    Full QVPO + Hy-Q training loop for continuous CartPole.

    Phase 1 — Offline pretraining
      Pure offline training on the demo dataset.
      Warms up the critics and diffusion policy before any environment interaction.

    Phase 2 — Online finetuning
      Collects experience with the current diffusion policy via the K_b-efficient
      behaviour policy (draw K_b candidates, act with argmax Q).
      Hy-Q mixer anneals offline ratio β from 1.0 → β_end.
    """

    def __init__(self, cfg: argparse.Namespace, device: torch.device):
        self.cfg    = cfg
        self.device = device

        # ── Buffers ──────────────────────────────────────────────────────────
        self.offline_buf = OfflineBuffer(cfg.demo_path, device, cfg.force_mag)
        self.online_buf  = OnlineBuffer(
            cfg.online_capacity, cfg.state_dim, cfg.action_dim, device
        )

        # ── Diffusion schedule ────────────────────────────────────────────────
        self.diffusion = GaussianDiffusion(
            n_steps=cfg.n_diffusion_steps,
            beta_min=cfg.beta_min,
            beta_max=cfg.beta_max,
        ).to(device)

        # ── Epsilon-prediction network ────────────────────────────────────────
        self.eps_net = EpsilonNet(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            time_emb_dim=cfg.time_emb_dim,
            n_steps=cfg.n_diffusion_steps,
        ).to(device)

        # ── Twin critics + frozen targets ─────────────────────────────────────
        self.q1        = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.q2        = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)

        # ── Optimisers ────────────────────────────────────────────────────────
        self.opt_eps = optim.Adam(self.eps_net.parameters(), lr=cfg.lr_policy)
        self.opt_q   = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr_q
        )

        # ── Hy-Q Mixer ────────────────────────────────────────────────────────
        self.mixer = HyQMixer(
            self.offline_buf, self.online_buf,
            beta_start=1.0,
            beta_end=cfg.hyq_beta_end,
            anneal_steps=cfg.hyq_anneal_steps,
            td_alpha=cfg.hyq_td_alpha,
        )

        self.log = {"critic_loss": [], "policy_loss": [], "episode_return": []}

    # ── Soft target update ─────────────────────────────────────────────────

    def _soft_update(self):
        tau = self.cfg.tau
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.lerp_(p.data, tau)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.lerp_(p.data, tau)

    # ── Critic update ──────────────────────────────────────────────────────

    def _critic_loss(self, batch: dict) -> Tuple[torch.Tensor, np.ndarray]:
        s, a, r, s_, d = (
            batch["states"], batch["actions"], batch["rewards"],
            batch["next_states"], batch["dones"],
        )
        cfg = self.cfg

        with torch.no_grad():
            # K_t-candidate target policy (QVPO §4.4)
            a_next    = self.diffusion.p_sample(self.eps_net, s_, n_samples=cfg.K_t)
            s_next_r  = s_.repeat_interleave(cfg.K_t, dim=0)
            q_next    = q_min(self.q1_target, self.q2_target, s_next_r, a_next)
            q_next    = q_next.view(-1, cfg.K_t).mean(dim=1)        # (B,)
            td_target = r + cfg.gamma * (1.0 - d) * q_next
            td_target = td_target.clamp(-200.0, 200.0)

        q1_pred = self.q1(s, a).squeeze(-1)
        q2_pred = self.q2(s, a).squeeze(-1)
        td_err  = ((td_target - q1_pred + td_target - q2_pred) / 2.0).detach().cpu().numpy()
        loss    = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)
        return loss, td_err

    # ── Policy (diffusion) update (QVPO Algorithm 1) ──────────────────────

    def _policy_loss(self, batch: dict) -> torch.Tensor:
        s   = batch["states"]
        B   = s.shape[0]
        cfg = self.cfg

        with torch.no_grad():
            # Sample Nd actions per state from current diffusion policy
            acts_nd  = self.diffusion.p_sample(self.eps_net, s, n_samples=cfg.Nd)
            s_rep    = s.repeat_interleave(cfg.Nd, dim=0)
            q_vals   = q_min(self.q1, self.q2, s_rep, acts_nd).view(B, cfg.Nd)

            # Advantage and qadv weights  ω_eq = max(A, 0)  (Eq. 9)
            v_s      = q_vals.mean(dim=1, keepdim=True)
            adv      = q_vals - v_s
            weights  = adv.clamp(min=0.0)

            # Best-advantage action per state
            best_idx = adv.argmax(dim=1)
            row_idx  = torch.arange(B, device=self.device)
            a_sel    = acts_nd.view(B, cfg.Nd, -1)[row_idx, best_idx]   # (B, action_dim)
            w_sel    = weights[row_idx, best_idx]                         # (B,)
            omega_s  = cfg.omega_ent * w_sel                             # (B,)

        # Q-weighted VLO loss  (Eq. 6)
        loss_q = self.diffusion.q_weighted_vlo_loss(self.eps_net, a_sel, s, w_sel)

        # Entropy regularisation  (Eq. 10)
        loss_e = self.diffusion.entropy_loss(
            self.eps_net, s, omega_s, cfg.Ne, cfg.force_mag
        )
        return loss_q + loss_e

    # ── Combined update step ───────────────────────────────────────────────

    def _update_step(self, batch: dict,
                     offline_idx: Optional[np.ndarray]) -> Tuple[float, float]:
        # Critic
        c_loss, td_err = self._critic_loss(batch)
        self.opt_q.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.opt_q.step()

        if offline_idx is not None:
            self.mixer.update_priorities(offline_idx, td_err[:len(offline_idx)])

        # Policy
        p_loss = self._policy_loss(batch)
        self.opt_eps.zero_grad()
        p_loss.backward()
        nn.utils.clip_grad_norm_(self.eps_net.parameters(), 1.0)
        self.opt_eps.step()

        self._soft_update()
        return c_loss.item(), p_loss.item()

    # ── K_b-efficient action selection (QVPO §4.4) ────────────────────────

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Draw K_b candidates from the diffusion policy; return argmax_Q.
        Output is a (action_dim,) numpy array in the raw force scale.
        """
        s          = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        candidates = self.diffusion.p_sample(self.eps_net, s, n_samples=self.cfg.K_b)
        s_rep      = s.expand(self.cfg.K_b, -1)
        q_vals     = q_min(self.q1, self.q2, s_rep, candidates)
        best       = candidates[q_vals.argmax()]
        return best.cpu().numpy()                                   # (action_dim,)

    # ── Phase 1: Offline pretraining ──────────────────────────────────────

    def offline_pretrain(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  Phase 1 — Offline Pretraining ({cfg.offline_steps:,} steps)")
        print(f"{'='*60}")

        for step in range(1, cfg.offline_steps + 1):
            batch = self.offline_buf.sample(cfg.batch_size)

            c_loss, _ = self._critic_loss(batch)
            self.opt_q.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
            )
            self.opt_q.step()

            p_loss = self._policy_loss(batch)
            self.opt_eps.zero_grad()
            p_loss.backward()
            nn.utils.clip_grad_norm_(self.eps_net.parameters(), 1.0)
            self.opt_eps.step()
            self._soft_update()

            if step % cfg.log_interval == 0:
                print(f"  [offline {step:6d}/{cfg.offline_steps}]  "
                      f"critic={c_loss.item():.4f}  policy={p_loss.item():.4f}")

        print("  Offline pretraining done.\n")

    # ── Phase 2: Online finetuning ─────────────────────────────────────────

    def online_finetune(self, env: ContinuousCartPoleEnv):
        cfg = self.cfg
        print(f"{'='*60}")
        print(f"  Phase 2 — Online Finetuning ({cfg.online_steps:,} steps)")
        print(f"{'='*60}")

        state, _  = env.reset(seed=cfg.seed)
        ep_return = 0.0
        ep_count  = 0

        for step in range(1, cfg.online_steps + 1):
            action = self.select_action(state)
            ns, reward, term, trunc, _ = env.step(action)
            done   = term or trunc
            self.online_buf.add(state, action, reward, ns, float(done))
            state      = ns
            ep_return += reward

            if done:
                self.log["episode_return"].append(ep_return)
                ep_count += 1
                if ep_count % cfg.eval_interval_eps == 0:
                    avg = np.mean(self.log["episode_return"][-20:])
                    print(f"  [online {step:7d}/{cfg.online_steps}]  "
                          f"ep={ep_count:4d}  avg_return(20)={avg:6.1f}  "
                          f"β={self.mixer.beta:.3f}  online={self.online_buf.size}")
                ep_return = 0.0
                state, _ = env.reset()

            if self.online_buf.size < cfg.batch_size:
                continue

            batch, off_idx = self.mixer.sample(cfg.batch_size)
            c_loss, p_loss = self._update_step(batch, off_idx)
            self.log["critic_loss"].append(c_loss)
            self.log["policy_loss"].append(p_loss)

        print("  Online finetuning done.\n")

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, env: ContinuousCartPoleEnv, n_episodes: int = 10) -> float:
        returns = []
        for ep in range(n_episodes):
            state, _ = env.reset()
            ep_ret   = 0.0
            done     = False
            while not done:
                action = self.select_action(state)
                state, reward, term, trunc, _ = env.step(action)
                ep_ret += reward
                done    = term or trunc
            returns.append(ep_ret)
            print(f"    ep {ep+1:2d}: {ep_ret:.0f}")
        mean_ret = float(np.mean(returns))
        print(f"  → mean={mean_ret:.1f}  std={np.std(returns):.1f}")
        return mean_ret

    # ── Checkpoint ─────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "eps_net":   self.eps_net.state_dict(),
            "q1":        self.q1.state_dict(),
            "q2":        self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
        }, path)
        print(f"  Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.eps_net.load_state_dict(ckpt["eps_net"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        print(f"  Checkpoint loaded ← {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Configuration & Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def build_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Diffusion Q-Learning (QVPO + Hy-Q) — Continuous CartPole"
    )

    # Environment
    p.add_argument("--state_dim",   type=int,   default=4)
    p.add_argument("--action_dim",  type=int,   default=1)
    p.add_argument("--force_mag",   type=float, default=10.0,
                   help="Max force ±F applied to the cart (replaces bang-bang ±10N)")
    p.add_argument("--max_steps",   type=int,   default=500)
    p.add_argument("--seed",        type=int,   default=42)

    # Demo data
    p.add_argument("--demo_path",   default="cartpole_demo_data.npz")

    # Diffusion model
    p.add_argument("--n_diffusion_steps", type=int,   default=5,
                   help="DDPM chain length T  (paper: 5)")
    p.add_argument("--beta_min",          type=float, default=0.1)
    p.add_argument("--beta_max",          type=float, default=0.5)
    p.add_argument("--hidden_dim",        type=int,   default=256)
    p.add_argument("--time_emb_dim",      type=int,   default=16)

    # QVPO policy update
    p.add_argument("--Nd",         type=int,   default=64,
                   help="Policy samples per state for qadv weights  (paper: 64)")
    p.add_argument("--Ne",         type=int,   default=64,
                   help="Uniform samples per state for entropy term  (paper: 64)")
    p.add_argument("--omega_ent",  type=float, default=1.0,
                   help="Entropy regularisation coefficient  (paper: 1.0)")
    p.add_argument("--K_b",        type=int,   default=10,
                   help="Behaviour policy candidates  (paper: 10)")
    p.add_argument("--K_t",        type=int,   default=2,
                   help="Target policy candidates  (paper: 2)")

    # Critic / RL
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--tau",        type=float, default=0.005)
    p.add_argument("--lr_q",       type=float, default=3e-4)
    p.add_argument("--lr_policy",  type=float, default=3e-4)
    p.add_argument("--batch_size", type=int,   default=256)

    # Hy-Q
    p.add_argument("--hyq_beta_end",     type=float, default=0.25)
    p.add_argument("--hyq_anneal_steps", type=int,   default=50_000)
    p.add_argument("--hyq_td_alpha",     type=float, default=0.6)
    p.add_argument("--online_capacity",  type=int,   default=200_000)

    # Training schedule
    p.add_argument("--offline_steps",    type=int,   default=20_000)
    p.add_argument("--online_steps",     type=int,   default=100_000)
    p.add_argument("--log_interval",     type=int,   default=1_000)
    p.add_argument("--eval_interval_eps",type=int,   default=10)
    p.add_argument("--save_path",        default="checkpoints/diffusion_ql.pt")

    return p.parse_args()


def main():
    cfg    = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    set_seed(cfg.seed)

    env = ContinuousCartPoleEnv(
        force_mag=cfg.force_mag,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
    )

    trainer = DiffusionQLTrainer(cfg, device)

    # Phase 1 — offline pretraining
    trainer.offline_pretrain()
    trainer.save(cfg.save_path.replace(".pt", "_offline.pt"))

    print("\n  [Eval] After offline pretraining:")
    trainer.evaluate(env, n_episodes=10)

    # Phase 2 — online finetuning
    trainer.online_finetune(env)
    trainer.save(cfg.save_path)

    print("\n  [Eval] After online finetuning:")
    trainer.evaluate(env, n_episodes=20)

    env.close()


if __name__ == "__main__":
    main()
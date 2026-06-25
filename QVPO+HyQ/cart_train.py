"""
Diffusion Q-Learning for Continuous CartPole
=============================================
Combines:
  - QVPO  : Q-weighted VLO loss with Gaussian (continuous) diffusion policy
  - Hy-Q  : Hybrid offline + online Q-learning with importance-weighted mixing

Bugs fixed vs. the crashing version (document 4)
-------------------------------------------------
  BUG 1 — Shape mismatch: rewards/dones inconsistent between buffers
    OfflineBuffer stored rewards as (N,) but critic expected (B,1) for
    clean squeeze(-1).  OnlineBuffer stored rewards as (N,) too.
    Fix: both buffers now store rewards and dones as (N,1) consistently.
    HyQMixer.torch.cat then always produces (B,1) which squeeze(-1) → (B,).

  BUG 2 — DeprecationWarning / future crash in ContinuousCartPoleEnv.step()
    select_action() returns shape (action_dim,) = (1,).
    float(np.clip(array_shape_1,)) is deprecated in NumPy ≥1.25.
    Fix: use float(np.clip(action, ...).item()) or index [0] before float().

  BUG 3 — Policy never learns (stuck at return ~8)
    Two causes:
      (a) offline_steps=1000 is too few — critic loss grows 0.006→3.1
          monotonically, it never converges before online phase begins.
          The policy loss was ~0.13 throughout because qadv weights were
          near-zero (bad Q-values → all advantages near zero → max(A,0)≈0).
          Fix: default offline_steps raised to 20_000.
      (b) HyQMixer replace=False with p=probs on 655k buffer is O(N log N).
          Fix: replace=True (Vose alias method, O(N)).
"""

import os
import copy
import math
import random
import argparse
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from metrics import MetricsTracker


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
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousCartPoleEnv:
    """
    CartPole with a native continuous scalar force action.

    State  : [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    Action : scalar force f ∈ [-force_mag, +force_mag]
    Reward : +1.0 every step the pole stays up
    Done   : |pole| > 12° OR |cart| > 2.4 m OR steps ≥ max_steps
    """

    GRAVITY      = 9.8
    MASSCART     = 1.0
    MASSPOLE     = 0.1
    TOTAL_MASS   = MASSCART + MASSPOLE
    HALF_LEN     = 0.5
    POLEMASS_LEN = MASSPOLE * HALF_LEN
    TAU          = 0.02
    THETA_THRESHOLD = 12 * 2 * math.pi / 360
    X_THRESHOLD     = 2.4

    def __init__(self, force_mag: float = 10.0, max_steps: int = 500,
                 seed: int = 42):
        self.force_mag   = force_mag
        self.max_steps   = max_steps
        self._rng        = np.random.RandomState(seed)
        self.state       = None
        self._step_count = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.state       = self._rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        self._step_count = 0
        return self.state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # FIX BUG 2: action is shape (1,) from select_action — extract scalar safely
        f = float(np.asarray(action).flat[0])
        f = max(-self.force_mag, min(self.force_mag, f))   # pure-Python clip, no NumPy warning

        x, x_dot, theta, theta_dot = self.state
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        tmp       = (f + self.POLEMASS_LEN * theta_dot ** 2 * sin_t) / self.TOTAL_MASS
        theta_acc = (self.GRAVITY * sin_t - cos_t * tmp) / (
            self.HALF_LEN * (4.0 / 3.0 - self.MASSPOLE * cos_t ** 2 / self.TOTAL_MASS)
        )
        x_acc     = tmp - self.POLEMASS_LEN * theta_acc * cos_t / self.TOTAL_MASS
        x         += self.TAU * x_dot
        x_dot     += self.TAU * x_acc
        theta     += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc
        self.state        = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._step_count += 1
        terminated = bool(abs(x) > self.X_THRESHOLD or abs(theta) > self.THETA_THRESHOLD)
        truncated  = self._step_count >= self.max_steps
        reward     = 1.0 if not terminated else 0.0
        return self.state.copy(), reward, terminated, truncated, {}

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Replay Buffers
# ══════════════════════════════════════════════════════════════════════════════

class OfflineBuffer:
    """
    Loads the static demonstration dataset.

    FIX BUG 1: rewards and dones stored as (N,1) — consistent with OnlineBuffer.
    This ensures torch.cat in HyQMixer always yields (B,1) which critic
    safely squeezes to (B,) via .squeeze(-1).
    """

    def __init__(self, path: str, device: torch.device, force_mag: float = 10.0):
        raw        = np.load(path)
        states     = torch.tensor(raw["states"],      dtype=torch.float32)
        actions_d  = torch.tensor(raw["actions"],     dtype=torch.float32)
        rewards    = torch.tensor(raw["rewards"],     dtype=torch.float32)
        next_states= torch.tensor(raw["next_states"], dtype=torch.float32)

        # Discrete {0,1} → continuous {-force_mag, +force_mag}, shape (N,1)
        actions_c = ((actions_d * 2.0 - 1.0) * force_mag).unsqueeze(-1)

        # No dones in file — all transitions mid-episode
        dones = torch.zeros(len(states), dtype=torch.float32)

        self.states      = states.to(device)
        self.actions     = actions_c.to(device)                    # (N, 1)
        self.rewards     = rewards.unsqueeze(-1).to(device)        # (N, 1)  ← FIX
        self.next_states = next_states.to(device)
        self.dones       = dones.unsqueeze(-1).to(device)          # (N, 1)  ← FIX
        self.size        = len(states)
        self.device      = device

        print(f"[OfflineBuffer] {self.size:,} transitions | "
              f"actions: 0→{-force_mag:.1f} N,  1→{+force_mag:.1f} N")

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "states":      self.states[idx],
            "actions":     self.actions[idx],       # (B, 1)
            "rewards":     self.rewards[idx],        # (B, 1)
            "next_states": self.next_states[idx],
            "dones":       self.dones[idx],          # (B, 1)
        }


class OnlineBuffer:
    """
    GPU ring-buffer for online experience.

    FIX BUG 1: rewards and dones stored as (N,1) to match OfflineBuffer
    so HyQMixer.torch.cat always produces consistent (B,1) tensors.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 device: torch.device):
        self.capacity = capacity
        self.device   = device
        self._ptr     = 0
        self._full    = False

        self.states      = torch.zeros((capacity, state_dim),  device=device)
        self.actions     = torch.zeros((capacity, action_dim), device=device)
        self.rewards     = torch.zeros((capacity, 1),          device=device)  # ← FIX
        self.next_states = torch.zeros((capacity, state_dim),  device=device)
        self.dones       = torch.zeros((capacity, 1),          device=device)  # ← FIX

    @property
    def size(self) -> int:
        return self.capacity if self._full else self._ptr

    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, next_state: np.ndarray, done: float):
        i = self._ptr
        self.states[i]      = torch.as_tensor(state,      dtype=torch.float32, device=self.device)
        self.actions[i]     = torch.as_tensor(action,     dtype=torch.float32, device=self.device)
        self.rewards[i, 0]  = reward    # scalar into (N,1) ← FIX
        self.next_states[i] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[i, 0]    = done      # scalar into (N,1) ← FIX
        self._ptr = (self._ptr + 1) % self.capacity
        if self._ptr == 0:
            self._full = True

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "states":      self.states[idx],
            "actions":     self.actions[idx],
            "rewards":     self.rewards[idx],        # (B, 1)
            "next_states": self.next_states[idx],
            "dones":       self.dones[idx],          # (B, 1)
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Hy-Q Mixer
# ══════════════════════════════════════════════════════════════════════════════

class HyQMixer:
    """
    Hy-Q priority-weighted offline/online batch mixer.

    FIX BUG 3b: replace=True in np.random.choice.
      replace=False forces O(N log N) sort on 655k buffer per step.
      replace=True uses O(N) Vose alias method.
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
                self.offline.size, size=n_offline,
                replace=True,   # FIX BUG 3b: O(N) alias method vs O(N log N)
                p=probs
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
# 4.  Gaussian Diffusion
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
    """ε_θ(ã_t, s, t) → ε̂  — noise prediction MLP."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, time_emb_dim: int = 16, n_steps: int = 5):
        super().__init__()
        self.action_dim = action_dim
        self.time_emb   = SinusoidalTimeEmbedding(time_emb_dim)
        in_dim = action_dim + state_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),   nn.Mish(),
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
        return self.net(torch.cat([a_t, self.time_emb(t), state], dim=-1))


class GaussianDiffusion:
    """DDPM schedule + forward/reverse for continuous actions."""

    def __init__(self, n_steps: int = 5,
                 beta_min: float = 0.1, beta_max: float = 0.5):
        self.T         = n_steps
        betas          = torch.linspace(beta_min, beta_max, n_steps)
        alphas         = 1.0 - betas
        alpha_bar      = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        self.betas          = betas
        self.alpha_bar      = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_ab        = alpha_bar.sqrt()
        self.sqrt_1mab      = (1.0 - alpha_bar).sqrt()
        self.posterior_var  = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * betas

    def to(self, device: torch.device) -> "GaussianDiffusion":
        for attr in ["betas", "alpha_bar", "alpha_bar_prev",
                     "sqrt_ab", "sqrt_1mab", "posterior_var"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, a0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(a0)
        s_ab   = self.sqrt_ab[t - 1].view(-1, 1)
        s_1mab = self.sqrt_1mab[t - 1].view(-1, 1)
        return s_ab * a0 + s_1mab * noise, noise

    @torch.no_grad()
    def _p_step(self, eps_net: EpsilonNet, a_t: torch.Tensor,
                state: torch.Tensor, t_val: int) -> torch.Tensor:
        B       = a_t.shape[0]
        t       = torch.full((B,), t_val, dtype=torch.long, device=a_t.device)
        eps_hat = eps_net(a_t, state, t)
        s_ab    = self.sqrt_ab[t_val - 1]
        s_1mab  = self.sqrt_1mab[t_val - 1]
        a0_hat  = (a_t - s_1mab * eps_hat) / s_ab
        ab      = self.alpha_bar[t_val - 1]
        ab_prev = self.alpha_bar_prev[t_val - 1]
        beta_t  = self.betas[t_val - 1]
        mu = (ab_prev.sqrt() * beta_t * a0_hat +
              (1.0 - ab_prev) * (1.0 - beta_t).sqrt() * a_t) / (1.0 - ab)
        if t_val == 1:
            return mu
        return mu + self.posterior_var[t_val - 1].sqrt() * torch.randn_like(a_t)

    @torch.no_grad()
    def p_sample(self, eps_net: EpsilonNet, state: torch.Tensor,
                 n_samples: int = 1) -> torch.Tensor:
        """Sample n_samples actions per state. Returns (B*n_samples, action_dim)."""
        if n_samples > 1:
            state = state.repeat_interleave(n_samples, dim=0)
        a_t = torch.randn(state.shape[0], eps_net.action_dim, device=state.device)
        for t_val in reversed(range(1, self.T + 1)):
            a_t = self._p_step(eps_net, a_t, state, t_val)
        return a_t

    def q_weighted_vlo_loss(self, eps_net: EpsilonNet, a_sel: torch.Tensor,
                            state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """L(θ) = E[ω_eq(s,a) · ||ε − ε_θ(ã_t, s, t)||²]  (QVPO Eq. 6)"""
        B          = a_sel.shape[0]
        t          = torch.randint(1, self.T + 1, (B,), device=a_sel.device)
        a_t, noise = self.q_sample(a_sel, t)
        eps_hat    = eps_net(a_t, state, t)
        per_sample = ((noise - eps_hat) ** 2).mean(dim=-1)
        return (weights * per_sample).mean()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Twin Q-Network  Q(s, a) → scalar
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
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
        return self.net(torch.cat([state, action], dim=-1))


def q_min(q1: QNetwork, q2: QNetwork,
          state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return torch.min(q1(state, action), q2(state, action)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Trainer
# ══════════════════════════════════════════════════════════════════════════════

class DiffusionQLTrainer:
    """
    QVPO + Hy-Q training loop for continuous CartPole.

    Phase 1 — Offline pretraining (default 20,000 steps)
      FIX BUG 3a: was 1,000 — too few for critic to converge.
      Critic must reach a stable loss before policy updates are meaningful.

    Phase 2 — Online finetuning
      K_b-efficient behaviour policy + Hy-Q mixing.
    """

    def __init__(self, cfg: argparse.Namespace, device: torch.device):
        self.cfg    = cfg
        self.device = device

        self.offline_buf = OfflineBuffer(cfg.demo_path, device, cfg.force_mag)
        self.online_buf  = OnlineBuffer(
            cfg.online_capacity, cfg.state_dim, cfg.action_dim, device
        )

        self.diffusion = GaussianDiffusion(
            n_steps=cfg.n_diffusion_steps,
            beta_min=cfg.beta_min,
            beta_max=cfg.beta_max,
        ).to(device)

        self.eps_net = EpsilonNet(
            state_dim    = cfg.state_dim,
            action_dim   = cfg.action_dim,
            hidden_dim   = cfg.hidden_dim,
            time_emb_dim = cfg.time_emb_dim,
            n_steps      = cfg.n_diffusion_steps,
        ).to(device)

        self.q1        = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.q2        = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)

        self.opt_eps = optim.Adam(self.eps_net.parameters(), lr=cfg.lr_policy)
        self.opt_q   = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr_q
        )

        self.mixer = HyQMixer(
            self.offline_buf, self.online_buf,
            beta_start=1.0, beta_end=cfg.hyq_beta_end,
            anneal_steps=cfg.hyq_anneal_steps, td_alpha=cfg.hyq_td_alpha,
        )

        # Persistent eval env (created once, never reopened in the loop)
        self._eval_env = ContinuousCartPoleEnv(
            force_mag=cfg.force_mag, max_steps=cfg.max_steps, seed=cfg.seed + 999
        )

        self.tracker = MetricsTracker("QVPO+HyQ")
        self.log = {"critic_loss": [], "policy_loss": [], "episode_return": []}

    # ── Soft update ────────────────────────────────────────────────────────

    def _soft_update(self):
        tau = self.cfg.tau
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.lerp_(p.data, tau)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.lerp_(p.data, tau)

    # ── Critic loss ────────────────────────────────────────────────────────

    def _critic_loss(self, batch: dict) -> Tuple[torch.Tensor, np.ndarray]:
        s  = batch["states"]
        a  = batch["actions"]
        r  = batch["rewards"].squeeze(-1)    # (B,1) → (B,)
        s_ = batch["next_states"]
        d  = batch["dones"].squeeze(-1)      # (B,1) → (B,)
        cfg = self.cfg

        with torch.no_grad():
            a_next   = self.diffusion.p_sample(self.eps_net, s_, n_samples=cfg.K_t)
            s_next_r = s_.repeat_interleave(cfg.K_t, dim=0)
            q_next   = q_min(self.q1_target, self.q2_target, s_next_r, a_next)
            q_next   = q_next.view(-1, cfg.K_t).mean(dim=1)
            td_target = (r + cfg.gamma * (1.0 - d) * q_next).clamp(-500.0, 500.0)

        q1_pred = self.q1(s, a).squeeze(-1)
        q2_pred = self.q2(s, a).squeeze(-1)
        td_err  = ((td_target - q1_pred + td_target - q2_pred) / 2.0
                   ).detach().cpu().numpy()
        loss    = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)
        return loss, td_err

    # ── Policy loss ────────────────────────────────────────────────────────

    def _policy_loss(self, batch: dict) -> torch.Tensor:
        s   = batch["states"]
        B   = s.shape[0]
        cfg = self.cfg

        with torch.no_grad():
            acts_nd = self.diffusion.p_sample(self.eps_net, s, n_samples=cfg.Nd)
            s_rep   = s.repeat_interleave(cfg.Nd, dim=0)
            q_vals  = q_min(self.q1, self.q2, s_rep, acts_nd).view(B, cfg.Nd)

            v_s     = q_vals.mean(dim=1, keepdim=True)
            adv     = q_vals - v_s
            weights = adv.clamp(min=0.0)

            best_idx = adv.argmax(dim=1)
            row_idx  = torch.arange(B, device=self.device)
            a_sel    = acts_nd.view(B, cfg.Nd, -1)[row_idx, best_idx]
            w_sel    = weights[row_idx, best_idx]

        return self.diffusion.q_weighted_vlo_loss(self.eps_net, a_sel, s, w_sel)

    # ── Combined update ────────────────────────────────────────────────────

    def _update_step(self, batch: dict,
                     offline_idx: Optional[np.ndarray]) -> Tuple[float, float]:
        c_loss, td_err = self._critic_loss(batch)
        self.opt_q.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.opt_q.step()

        if offline_idx is not None:
            self.mixer.update_priorities(offline_idx, td_err[:len(offline_idx)])

        p_loss = self._policy_loss(batch)
        self.opt_eps.zero_grad()
        p_loss.backward()
        nn.utils.clip_grad_norm_(self.eps_net.parameters(), 1.0)
        self.opt_eps.step()
        self._soft_update()
        return c_loss.item(), p_loss.item()

    # ── K_b-efficient action selection ────────────────────────────────────

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        s          = torch.tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        candidates = self.diffusion.p_sample(self.eps_net, s, n_samples=self.cfg.K_b)
        s_rep      = s.expand(self.cfg.K_b, -1)
        q_vals     = q_min(self.q1, self.q2, s_rep, candidates)
        return candidates[q_vals.argmax()].cpu().numpy()   # (action_dim,)

    # ── Evaluation using persistent env ───────────────────────────────────

    def _run_eval(self, step: int) -> float:
        returns: List[float] = []
        for ep in range(self.cfg.eval_episodes):
            s, _ = self._eval_env.reset(seed=self.cfg.seed + ep)
            ep_ret, done = 0.0, False
            while not done:
                a = self.select_action(s)
                s, r, term, trunc, _ = self._eval_env.step(a)
                ep_ret += r
                done = term or trunc
            returns.append(ep_ret)
        self.tracker.log_eval(step=step, returns=returns)
        avg = float(np.mean(returns))
        print(f"  [eval  step={step:7d}]  "
              f"mean={avg:6.1f}  std={float(np.std(returns)):5.1f}  "
              f"β={self.mixer.beta:.3f}  online={self.online_buf.size}")
        return avg

    # ── Phase 1: Offline pretraining ──────────────────────────────────────

    def offline_pretrain(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  Phase 1 — Offline Pretraining ({cfg.offline_steps:,} steps)")
        print(f"  (FIX: 20,000 steps so critic converges before online phase)")
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

            self.tracker.log_step(step=step, critic_loss=c_loss.item(),
                                  policy_loss=p_loss.item())

            if step % cfg.log_interval == 0:
                print(f"  [offline {step:6d}/{cfg.offline_steps}]  "
                      f"critic={c_loss.item():.4f}  policy={p_loss.item():.6f}")

        print("  Offline pretraining done.\n")

    # ── Phase 2: Online finetuning ─────────────────────────────────────────

    def online_finetune(self, env: ContinuousCartPoleEnv):
        cfg = self.cfg
        print(f"{'='*60}")
        print(f"  Phase 2 — Online Finetuning ({cfg.online_steps:,} steps)")
        print(f"{'='*60}")

        state, _ = env.reset(seed=cfg.seed)
        ep_return = 0.0

        for step in range(1, cfg.online_steps + 1):
            action = self.select_action(state)
            ns, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            self.online_buf.add(state, action, reward, ns, float(done))
            state      = ns
            ep_return += reward

            if done:
                self.log["episode_return"].append(ep_return)
                ep_return = 0.0
                state, _ = env.reset()

            if self.online_buf.size < cfg.batch_size:
                continue

            batch, off_idx = self.mixer.sample(cfg.batch_size)
            c_loss, p_loss = self._update_step(batch, off_idx)
            self.log["critic_loss"].append(c_loss)
            self.log["policy_loss"].append(p_loss)

            # Accumulate for tracker (flush every log_interval)
            if step % cfg.log_interval == 0:
                avg_c = float(np.mean(self.log["critic_loss"][-cfg.log_interval:]))
                avg_p = float(np.mean(self.log["policy_loss"][-cfg.log_interval:]))
                self.tracker.log_step(step=step, critic_loss=avg_c, policy_loss=avg_p)

            # Pure step-based eval trigger
            if step % cfg.eval_interval == 0:
                self._run_eval(step)

        self.tracker.save("results/qvpo+hy-q_cartpole_metrics.npz")
        print("  Online finetuning done.\n")

    # ── Final evaluation ───────────────────────────────────────────────────

    def evaluate(self, env: ContinuousCartPoleEnv, n_episodes: int = 10) -> float:
        returns = []
        for ep in range(n_episodes):
            state, _ = env.reset()
            ep_ret, done = 0.0, False
            while not done:
                action = self.select_action(state)
                state, reward, term, trunc, _ = env.step(action)
                ep_ret += reward
                done = term or trunc
            returns.append(ep_ret)
            print(f"    ep {ep+1:2d}: {ep_ret:.0f}")
        mean_ret = float(np.mean(returns))
        print(f"  → mean={mean_ret:.1f}  std={float(np.std(returns)):.1f}")
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

    def close(self):
        self._eval_env.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Configuration & Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def build_config() -> argparse.Namespace:
    p = argparse.ArgumentParser("QVPO + Hy-Q — Continuous CartPole")

    p.add_argument("--state_dim",   type=int,   default=4)
    p.add_argument("--action_dim",  type=int,   default=1)
    p.add_argument("--force_mag",   type=float, default=10.0)
    p.add_argument("--max_steps",   type=int,   default=500)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--demo_path",   default="cartpole_demo_data.npz")

    p.add_argument("--n_diffusion_steps", type=int,   default=5)
    p.add_argument("--beta_min",          type=float, default=0.1)
    p.add_argument("--beta_max",          type=float, default=0.5)
    p.add_argument("--hidden_dim",        type=int,   default=256)
    p.add_argument("--time_emb_dim",      type=int,   default=16)

    p.add_argument("--Nd",         type=int,   default=64)
    p.add_argument("--K_b",        type=int,   default=10)
    p.add_argument("--K_t",        type=int,   default=2)

    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--tau",        type=float, default=0.005)
    p.add_argument("--lr_q",       type=float, default=3e-4)
    p.add_argument("--lr_policy",  type=float, default=3e-4)
    p.add_argument("--batch_size", type=int,   default=256)

    p.add_argument("--hyq_beta_end",     type=float, default=0.25)
    p.add_argument("--hyq_anneal_steps", type=int,   default=50_000)
    p.add_argument("--hyq_td_alpha",     type=float, default=0.6)
    p.add_argument("--online_capacity",  type=int,   default=200_000)

    # FIX BUG 3a: raised from 1_000 to 20_000
    p.add_argument("--offline_steps",  type=int, default=20_000)
    p.add_argument("--online_steps",   type=int, default=100_000)
    p.add_argument("--log_interval",   type=int, default=1_000)
    p.add_argument("--eval_interval",  type=int, default=5_000)
    p.add_argument("--eval_episodes",  type=int, default=10)
    p.add_argument("--save_path",      default="checkpoints/diffusion_ql.pt")

    return p.parse_args()


def main():
    cfg    = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    set_seed(cfg.seed)

    os.makedirs("results", exist_ok=True)
    env     = ContinuousCartPoleEnv(force_mag=cfg.force_mag,
                                    max_steps=cfg.max_steps, seed=cfg.seed)
    trainer = DiffusionQLTrainer(cfg, device)

    trainer.offline_pretrain()
    trainer.save(cfg.save_path.replace(".pt", "_offline.pt"))

    print("\n  [Eval] After offline pretraining:")
    trainer.evaluate(env, n_episodes=10)

    trainer.online_finetune(env)
    trainer.save(cfg.save_path)

    print("\n  [Eval] Final:")
    trainer.evaluate(env, n_episodes=20)

    env.close()
    trainer.close()


if __name__ == "__main__":
    main()
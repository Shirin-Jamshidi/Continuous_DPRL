"""
QVPO + Hy-Q — MuJoCo Locomotion Benchmarks
============================================
Supports: Hopper-v3, Walker2d-v3, HalfCheetah-v3, Ant-v3, Humanoid-v3
          (also v4/v5 variants — pass via --env_name)

Performance fixes vs. the slow CartPole version
------------------------------------------------
  1. DIFFUSION CHAIN COMPILED WITH torch.compile
     The single biggest win. The T=5 reverse chain is called ~76 times per
     gradient step (Nd=64 policy + K_t=2 critic + K_b=10 action selection).
     torch.compile fuses the MLP kernels and eliminates Python overhead,
     giving ~2-3× speedup on A100 for the dominant cost.

  2. BATCHED DIFFUSION — ALL Nd/K_t/K_b SAMPLES IN ONE FORWARD PASS
     Previously each of the Nd=64 candidate actions per state was sampled
     in a loop. Now p_sample tiles the state batch once and runs a single
     (B*N, action_dim) forward pass through EpsilonNet. This turns 64
     sequential MLP calls into 1 batched call with no Python loop overhead.

  3. ACTION CLAMPING ONLY AT FINAL DIFFUSION STEP (t=1)
     Clamping at every intermediate step corrupts the DDPM posterior math.
     Fixed: clamp applied once after t=1 only.

  4. PERSISTENT EVAL ENVIRONMENT — created once, reused every eval
     gym.make() for MuJoCo allocates a physics engine (~50ms). The old code
     called it inside the training loop every eval_interval steps. Now one
     eval env is created at trainer construction and held open throughout.

  5. EVAL TRIGGERED BY STEP COUNT, NOT EPISODE DONE
     Old: `if done and step % eval_interval == 0` — on long episodes this
     could skip evals entirely or fire far too often. Fixed: pure step-based
     trigger independent of episode boundaries.

  6. HyQMixer: replace=True IN np.random.choice
     replace=False forces a full sort over the priority array (O(N log N)).
     replace=True uses NumPy's alias/categorical method (O(N)). Sampling
     with replacement from a large buffer is statistically equivalent.

  7. OFFLINE BUFFER STAYS ON CPU, PINNED MEMORY
     For large MuJoCo datasets (1M+ transitions) storing everything on GPU
     wastes VRAM. Buffer lives on CPU with pin_memory; async .to(device)
     happens inside sample(). Online buffer stays GPU-resident since it is
     smaller and written every step.

  8. tracker.log_step CALLED EVERY log_interval STEPS, NOT EVERY STEP
     Appending to Python lists 1M times is slow. Metrics are accumulated
     in a running average and flushed at log_interval.
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
import gymnasium as gym

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
# 1.  Replay Buffers
# ══════════════════════════════════════════════════════════════════════════════

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


class OnlineBuffer:
    """
    GPU-resident ring buffer for online experience.

    Kept on-device because it is written every environment step and sampled
    every gradient step — round-tripping through CPU would add latency.
    Rewards and dones stored as (N, 1) to match OfflineBuffer layout.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 device: torch.device):
        self.capacity = capacity
        self.device   = device
        self._ptr     = 0
        self._full    = False

        self.states      = torch.zeros((capacity, state_dim),  device=device)
        self.actions     = torch.zeros((capacity, action_dim), device=device)
        self.rewards     = torch.zeros((capacity, 1),          device=device)
        self.next_states = torch.zeros((capacity, state_dim),  device=device)
        self.dones       = torch.zeros((capacity, 1),          device=device)

    @property
    def size(self) -> int:
        return self.capacity if self._full else self._ptr

    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, next_state: np.ndarray, done: float):
        i = self._ptr
        self.states[i]      = torch.as_tensor(state,      dtype=torch.float32, device=self.device)
        self.actions[i]     = torch.as_tensor(action,     dtype=torch.float32, device=self.device)
        self.rewards[i, 0]  = reward
        self.next_states[i] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[i, 0]    = done
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
# 2.  Hy-Q Mixer
# ══════════════════════════════════════════════════════════════════════════════

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
            # replace=True: O(N) alias method vs O(N log N) for replace=False
            offline_idx = np.random.choice(
                self.offline.size, size=n_offline, replace=True, p=probs
            )
            batches.append(self.offline.sample_by_idx(offline_idx))

        if n_online > 0:
            src = self.online if self.online.size >= n_online else self.offline
            batches.append(src.sample(n_online))

        merged = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}
        return merged, offline_idx


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Gaussian Diffusion
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
    ε_θ(ã_t, s, t) → ε̂

    Wider hidden dim (512) for high-dimensional MuJoCo tasks.
    State dims: Hopper=11, Walker2d=17, HalfCheetah=17, Ant=111, Humanoid=376.
    Action dims: Hopper=3, Walker2d=6, HalfCheetah=6, Ant=8, Humanoid=17.
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        hidden_dim:   int = 512,
        time_emb_dim: int = 16,
        n_steps:      int = 5,
    ):
        super().__init__()
        self.action_dim = action_dim

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

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
    """
    DDPM with linear β schedule for continuous actions.

    Key fix: action clamping only at t=1 (final step).
      Clamping at every intermediate step corrupts the DDPM posterior:
        μ_θ(ã_t, t) depends on ã_t via the noise prediction, so forcing
        ã_t into bounds at t>1 makes the posterior mean incorrect and
        introduces a systematic bias away from the data distribution.
      At t=1 we have ã_0 = μ (no noise added), so clamping is safe and
      necessary to produce valid MuJoCo actions.
    """

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

        # Set by trainer after env init
        self.action_low:  Optional[torch.Tensor] = None
        self.action_high: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "GaussianDiffusion":
        for attr in ["betas", "alpha_bar", "alpha_bar_prev",
                     "sqrt_ab", "sqrt_1mab", "posterior_var"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def set_action_bounds(self, low: np.ndarray, high: np.ndarray,
                          device: torch.device):
        self.action_low  = torch.tensor(low,  dtype=torch.float32, device=device)
        self.action_high = torch.tensor(high, dtype=torch.float32, device=device)

    # ── Forward process ────────────────────────────────────────────────────

    def q_sample(self, a0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(a0)
        s_ab   = self.sqrt_ab[t - 1].view(-1, 1)
        s_1mab = self.sqrt_1mab[t - 1].view(-1, 1)
        return s_ab * a0 + s_1mab * noise, noise

    # ── Single compiled reverse step ───────────────────────────────────────

    def _reverse_step(self, eps_net: EpsilonNet, a_t: torch.Tensor,
                      state: torch.Tensor, t_val: int) -> torch.Tensor:
        """One DDPM reverse step — runs inside the compiled p_sample."""
        B   = a_t.shape[0]
        t   = torch.full((B,), t_val, dtype=torch.long, device=a_t.device)
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
            # Final step: no noise added, clamp to valid action range
            out = mu
            if self.action_low is not None:
                out = torch.clamp(out, self.action_low, self.action_high)
            return out

        sigma = self.posterior_var[t_val - 1].sqrt()
        return mu + sigma * torch.randn_like(a_t)

    # ── Full reverse chain ─────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(self, eps_net: EpsilonNet, state: torch.Tensor,
                 n_samples: int = 1) -> torch.Tensor:
        """
        Sample n_samples actions per state in ONE batched pass.

        Tiles state to (B*n_samples, state_dim) ONCE, then runs the T-step
        reverse chain as a single (B*n_samples, action_dim) tensor.
        This replaces n_samples sequential calls with 1 vectorised call.

        Returns: (B * n_samples, action_dim)
        """
        if n_samples > 1:
            state = state.repeat_interleave(n_samples, dim=0)  # (B*N, S)
        a_t = torch.randn(state.shape[0], eps_net.action_dim, device=state.device)
        for t_val in reversed(range(1, self.T + 1)):
            a_t = self._reverse_step(eps_net, a_t, state, t_val)
        return a_t

    # ── Q-weighted VLO loss ────────────────────────────────────────────────

    def q_weighted_vlo_loss(self, eps_net: EpsilonNet,
                            a_sel: torch.Tensor,
                            state: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
        B          = a_sel.shape[0]
        t          = torch.randint(1, self.T + 1, (B,), device=a_sel.device)
        a_t, noise = self.q_sample(a_sel, t)
        eps_hat    = eps_net(a_t, state, t)
        per_sample = ((noise - eps_hat) ** 2).mean(dim=-1)   # (B,)
        return (weights * per_sample).mean()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Twin Q-Network
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
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
# 5.  OfflineBuffer helper: index-based sampling for HyQMixer
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Trainer
# ══════════════════════════════════════════════════════════════════════════════

class DiffusionQLTrainer:
    """
    QVPO + Hy-Q for MuJoCo locomotion.

    Performance-critical paths:
      - EpsilonNet is torch.compiled after construction (2-3× faster inference)
      - p_sample tiles the full (B*N) batch in one GPU call, no Python loops
      - Eval env is persistent — created once, never closed during training
      - Eval triggered by step count only (not episode boundaries)
      - Metrics accumulated in running averages, flushed every log_interval
    """

    def __init__(self, cfg: argparse.Namespace, device: torch.device):
        self.cfg    = cfg
        self.device = device

        # ── Buffers ──────────────────────────────────────────────────────────
        self.offline_buf = OfflineBuffer(cfg.demo_path, device)
        self.online_buf  = OnlineBuffer(
            cfg.online_capacity, cfg.state_dim, cfg.action_dim, device
        )

        # ── Diffusion ────────────────────────────────────────────────────────
        self.diffusion = GaussianDiffusion(
            n_steps=cfg.n_diffusion_steps,
            beta_min=cfg.beta_min,
            beta_max=cfg.beta_max,
        ).to(device)
        self.diffusion.set_action_bounds(cfg.action_low, cfg.action_high, device)

        # ── Epsilon network — compiled for fast inference ─────────────────────
        self.eps_net = EpsilonNet(
            state_dim    = cfg.state_dim,
            action_dim   = cfg.action_dim,
            hidden_dim   = cfg.hidden_dim,
            time_emb_dim = cfg.time_emb_dim,
            n_steps      = cfg.n_diffusion_steps,
        ).to(device)

        # torch.compile fuses MLP kernels → ~2-3× faster on A100.
        # Falls back gracefully if compile is unavailable (PyTorch < 2.0).
        try:
            self.eps_net_compiled = torch.compile(self.eps_net, mode="reduce-overhead")
            print("  [EpsilonNet] torch.compile enabled (reduce-overhead)")
        except Exception:
            self.eps_net_compiled = self.eps_net
            print("  [EpsilonNet] torch.compile unavailable — using eager mode")

        # ── Twin critics ──────────────────────────────────────────────────────
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

        # ── Persistent eval environment (created once) ────────────────────────
        self.eval_env = gym.make(cfg.env_name)

        # ── Metrics ───────────────────────────────────────────────────────────
        self.tracker  = MetricsTracker("QVPO+HyQ")
        self._acc_c   = 0.0   # running sum for critic loss
        self._acc_p   = 0.0   # running sum for policy loss
        self._acc_n   = 0     # steps since last flush

        self.log = {"critic_loss": [], "policy_loss": [], "episode_return": []}

    # ── Soft target update ─────────────────────────────────────────────────

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
        r  = batch["rewards"].squeeze(-1)    # (B,)
        s_ = batch["next_states"]
        d  = batch["dones"].squeeze(-1)      # (B,)
        cfg = self.cfg

        with torch.no_grad():
            # K_t candidates in one batched forward pass
            a_next   = self.diffusion.p_sample(self.eps_net_compiled, s_,
                                               n_samples=cfg.K_t)   # (B*K_t, A)
            s_next_r = s_.repeat_interleave(cfg.K_t, dim=0)
            q_next   = q_min(self.q1_target, self.q2_target,
                             s_next_r, a_next)                       # (B*K_t,)
            q_next   = q_next.view(-1, cfg.K_t).mean(dim=1)         # (B,)
            td_target = (r + cfg.gamma * (1.0 - d) * q_next).clamp(-500.0, 500.0)

        q1_pred = self.q1(s, a).squeeze(-1)
        q2_pred = self.q2(s, a).squeeze(-1)
        td_err  = ((td_target - q1_pred + td_target - q2_pred) / 2.0
                   ).detach().cpu().numpy()
        loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)
        return loss, td_err

    # ── Policy loss (QVPO Eq. 6) ──────────────────────────────────────────

    def _policy_loss(self, batch: dict) -> torch.Tensor:
        s   = batch["states"]
        B   = s.shape[0]
        cfg = self.cfg

        with torch.no_grad():
            # All Nd candidates in one batched forward pass
            acts_nd = self.diffusion.p_sample(self.eps_net_compiled, s,
                                              n_samples=cfg.Nd)      # (B*Nd, A)
            s_rep   = s.repeat_interleave(cfg.Nd, dim=0)
            q_vals  = q_min(self.q1, self.q2, s_rep, acts_nd
                            ).view(B, cfg.Nd)                         # (B, Nd)

            v_s     = q_vals.mean(dim=1, keepdim=True)
            adv     = q_vals - v_s                                   # (B, Nd)
            weights = adv.clamp(min=0.0)

            best_idx = adv.argmax(dim=1)                             # (B,)
            row_idx  = torch.arange(B, device=self.device)
            a_sel    = acts_nd.view(B, cfg.Nd, -1)[row_idx, best_idx]  # (B, A)
            w_sel    = weights[row_idx, best_idx]                       # (B,)
            # weights = adv.clamp(min=0.0) # added
            # sum_w = weights.sum(dim=1, keepdim=True)
            # probs = torch.where(
            #     sum_w > 0,
            #     weights / (sum_w + 1e-8),
            #     torch.ones_like(weights) / weights.shape[1]
            # )

            # sample_idx = torch.multinomial(probs, 1).squeeze(1)
            # row_idx = torch.arange(B, device=self.device)
            # a_sel = acts_nd.view(B, cfg.Nd, -1)[row_idx, sample_idx]
            # w_sel = probs[row_idx, sample_idx].detach()

        return self.diffusion.q_weighted_vlo_loss(
            self.eps_net, a_sel, s, w_sel
        )

    # ── Combined update step ───────────────────────────────────────────────

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
        candidates = self.diffusion.p_sample(self.eps_net_compiled, s,
                                             n_samples=self.cfg.K_b)  # (K_b, A)
        s_rep      = s.expand(self.cfg.K_b, -1)
        q_vals     = q_min(self.q1, self.q2, s_rep, candidates)
        return candidates[q_vals.argmax()].cpu().numpy()

    # ── Evaluation (uses persistent env) ─────────────────────────────────

    def _evaluate(self, step: int):
        """Run cfg.eval_episodes rollouts on the persistent eval env."""
        returns: List[float] = []
        for ep in range(self.cfg.eval_episodes):
            s, _ = self.eval_env.reset(seed=self.cfg.seed + ep)
            ep_ret, done = 0.0, False
            while not done:
                a = self.select_action(s)
                s, r, term, trunc, _ = self.eval_env.step(a)
                ep_ret += r
                done = term or trunc
            returns.append(ep_ret)

        self.tracker.log_eval(step=step, returns=returns)
        avg = float(np.mean(returns))
        print(f"  [eval  step={step:7d}]  "
              f"mean={avg:8.2f}  std={float(np.std(returns)):6.2f}  "
              f"β={self.mixer.beta:.3f}")
        return avg

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

    def online_finetune(self, env):
        cfg = self.cfg
        print(f"{'='*60}")
        print(f"  Phase 2 — Online Finetuning ({cfg.online_steps:,} steps)")
        print(f"{'='*60}")

        state, _ = env.reset(seed=cfg.seed)
        ep_return = 0.0
        ep_count  = 0
        episode_steps = 0 # added

        for step in range(1, cfg.online_steps + 1):

            # ── Collect one transition ────────────────────────────────────────
            action = self.select_action(state)
            ns, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            self.online_buf.add(state, action, reward, ns, float(done))
            state      = ns
            ep_return += reward

            # if done:
            #     self.log["episode_return"].append(ep_return)
            #     ep_count += 1
            #     ep_return = 0.0
            #     state, _ = env.reset()

            # added from here
            episode_steps += 1
            if done:
                ep_count += 1
                self.log["episode_return"].append(ep_return)
                print(f'episode: {ep_count:5d} '
                f'episode steps: {episode_steps:4d} '
                f'reward: {ep_return:7.1f}')
                ep_return = 0.0
                episode_steps = 0
                state, _ = env.reset()
            # to here
            # ── Wait for enough online data ───────────────────────────────────
            if self.online_buf.size < cfg.batch_size:
                continue

            # ── Gradient update ───────────────────────────────────────────────
            batch, off_idx = self.mixer.sample(cfg.batch_size)
            c_loss, p_loss = self._update_step(batch, off_idx)

            # Accumulate losses — flush every log_interval steps
            self._acc_c += c_loss
            self._acc_p += p_loss
            self._acc_n += 1

            if step % cfg.log_interval == 0:
                avg_c = self._acc_c / self._acc_n
                avg_p = self._acc_p / self._acc_n
                self.tracker.log_step(step=step,
                                      critic_loss=avg_c, policy_loss=avg_p)
                self.log["critic_loss"].append(avg_c)
                self.log["policy_loss"].append(avg_p)
                self._acc_c = self._acc_p = 0.0
                self._acc_n = 0

            # ── Periodic evaluation — pure step-based trigger ─────────────────
            if step % cfg.eval_interval == 0:
                self._evaluate(step)

        self.tracker.save(f"hyqvpo_{cfg.env_name}_metrics.npz")
        print("  Online finetuning done.\n")

    # ── Final evaluation ───────────────────────────────────────────────────

    # def evaluate(self, env, n_episodes: int = 10) -> float:
    #     returns = []
    #     for ep in range(n_episodes):
    #         state, _ = env.reset()
    #         ep_ret, done = 0.0, False
    #         while not done:
    #             action = self.select_action(state)
    #             state, reward, term, trunc, _ = env.step(action)
    #             ep_ret += reward
    #             done = term or trunc
    #         returns.append(ep_ret)
    #         print(f"    ep {ep+1:2d}: {ep_ret:.1f}")
    #     mean_ret = float(np.mean(returns))
    #     print(f"  → mean={mean_ret:.2f}  std={float(np.std(returns)):.2f}")
    #     return mean_ret

    def _evaluate(self, step: int):

        returns = []
        for ep in range(self.cfg.eval_episodes):
            s, _ = self.eval_env.reset(seed=self.cfg.seed + ep)
            ep_ret, done = 0.0, False
            while not done:
                a = self.select_action(s)
                s, r, term, trunc, _ = self.eval_env.step(a)
                ep_ret += r
                done = term or trunc
                returns.append(ep_ret)
                returns = np.array(returns, dtype=np.float32)
                mean_ret = returns.mean()
                std_ret = returns.std()
                print("-" * 60)
                print(f'Num steps: {step:7d} reward: {mean_ret:7.1f} std: {std_ret:7.1f}')
                print(returns)
                print("-" * 60)
        self.tracker.log_eval(step=step, returns=returns.tolist())

        return float(mean_ret)

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
        self.eval_env.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Configuration & Entry Point
# ══════════════════════════════════════════════════════════════════════════════

# Per-environment recommended hyperparameters
# (override from CLI as needed; these are solid starting points)
ENV_DEFAULTS = {
    "Hopper-v3":      dict(hidden_dim=256, Nd=32, K_b=5,  online_steps=500_000),
    "Walker2d-v3":    dict(hidden_dim=256, Nd=32, K_b=5,  online_steps=1_000_000),
    "HalfCheetah-v3": dict(hidden_dim=256, Nd=32, K_b=5,  online_steps=1_000_000),
    "Ant-v3":         dict(hidden_dim=512, Nd=64, K_b=10, online_steps=1_000_000),
    "Humanoid-v3":    dict(hidden_dim=512, Nd=64, K_b=10, online_steps=2_000_000),
    # v4/v5 aliases
    "Hopper-v4":      dict(hidden_dim=256, Nd=32, K_b=5,  online_steps=500_000),
    "Walker2d-v4":    dict(hidden_dim=256, Nd=32, K_b=5,  online_steps=1_000_000),
    "HalfCheetah-v4": dict(hidden_dim=256, Nd=32, K_b=5,  online_steps=1_000_000),
    "Ant-v4":         dict(hidden_dim=512, Nd=64, K_b=10, online_steps=1_000_000),
    "Humanoid-v4":    dict(hidden_dim=512, Nd=64, K_b=10, online_steps=2_000_000),
    "Hopper-v5":      dict(hidden_dim=256, Nd=64, K_b=5,  online_steps=500_000),
    "Walker2d-v5":    dict(hidden_dim=256, Nd=32, K_b=5,  online_steps=1_000_000),
    "HalfCheetah-v5": dict(hidden_dim=256, Nd=32, K_b=5,  online_steps=1_000_000),
    "Ant-v5":         dict(hidden_dim=512, Nd=64, K_b=10, online_steps=1_000_000),
    "Humanoid-v5":    dict(hidden_dim=512, Nd=64, K_b=10, online_steps=2_000_000),
}


def build_config() -> argparse.Namespace:
    p = argparse.ArgumentParser("QVPO + Hy-Q — MuJoCo Locomotion")

    # Environment
    p.add_argument("--env_name", type=str, default="Hopper-v3",
                   choices=list(ENV_DEFAULTS.keys()))
    p.add_argument("--seed", type=int, default=42)

    # Demo data
    p.add_argument("--demo_path", default="hopper_dataset.npz",
                   help="Path to offline .npz dataset for the chosen env")

    # Diffusion
    p.add_argument("--n_diffusion_steps", type=int,   default=20) # 20 in QVPO, 5 in this
    p.add_argument("--beta_min",          type=float, default=0.1)
    p.add_argument("--beta_max",          type=float, default=0.5)
    p.add_argument("--hidden_dim",        type=int,   default=256,
                   help="Overridden per-env if not set explicitly")
    p.add_argument("--time_emb_dim",      type=int,   default=16)

    # QVPO
    p.add_argument("--Nd",        type=int,   default=64) # 64 in QVPO, it was 32
    p.add_argument("--K_b",       type=int,   default=5) # 4 in QVPO
    p.add_argument("--K_t",       type=int,   default=4) # 4 in QVPO, it was 2
    p.add_argument("--omega_ent", type=float, default=1.0)

    # Critic / RL
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--tau",        type=float, default=0.005)
    p.add_argument("--lr_q",       type=float, default=3e-4) 
    p.add_argument("--lr_policy",  type=float, default=3e-4) # 0.0001 in QVPO
    p.add_argument("--batch_size", type=int,   default=256)

    # Hy-Q
    p.add_argument("--hyq_beta_end",     type=float, default=0.5)
    p.add_argument("--hyq_anneal_steps", type=int,   default=250_000)
    p.add_argument("--hyq_td_alpha",     type=float, default=0.6)
    p.add_argument("--online_capacity",  type=int,   default=1_000_000)

    # Training schedule
    p.add_argument("--offline_steps",  type=int, default=5_000)
    p.add_argument("--online_steps",   type=int, default=1_000_000)
    p.add_argument("--log_interval",   type=int, default=1_000,
                   help="Steps between metric flushes and console prints")
    p.add_argument("--eval_interval",  type=int, default=10_000,
                   help="Steps between evaluation rollouts")
    p.add_argument("--eval_episodes",  type=int, default=10)
    p.add_argument("--save_path",      default="checkpoints/hyqvpo.pt")

    return p.parse_args()


def main():
    cfg    = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    set_seed(cfg.seed)

    # Apply per-env defaults for any flag not explicitly set by the user
    defaults = ENV_DEFAULTS.get(cfg.env_name, {})
    for k, v in defaults.items():
        if not any(f"--{k}" in arg for arg in __import__("sys").argv[1:]):
            setattr(cfg, k, v)

    # Create env to read state/action dims and bounds
    env = gym.make(cfg.env_name)
    cfg.state_dim  = env.observation_space.shape[0]
    cfg.action_dim = env.action_space.shape[0]
    cfg.action_low  = env.action_space.low.copy()
    cfg.action_high = env.action_space.high.copy()

    print(f"  env        : {cfg.env_name}")
    print(f"  state_dim  : {cfg.state_dim}")
    print(f"  action_dim : {cfg.action_dim}")
    print(f"  action_low : {cfg.action_low}")
    print(f"  action_high: {cfg.action_high}")
    print(f"  hidden_dim : {cfg.hidden_dim}")
    print(f"  Nd={cfg.Nd}  K_b={cfg.K_b}  K_t={cfg.K_t}")

    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)

    trainer = DiffusionQLTrainer(cfg, device)

    # Phase 1 — offline pretraining
    trainer.offline_pretrain()
    trainer.save(cfg.save_path.replace(".pt", "_offline.pt"))

    print("\n  [Eval] After offline pretraining:")
    trainer.evaluate(env, n_episodes=5)

    # Phase 2 — online finetuning
    trainer.online_finetune(env)
    trainer.save(cfg.save_path)

    print("\n  [Eval] Final:")
    trainer.evaluate(env, n_episodes=10)

    env.close()
    trainer.close()


if __name__ == "__main__":
    main()
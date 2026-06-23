import gymnasium as gym   # ✅ NEW
import os
import copy
import math
import random
import argparse
from typing import Tuple, Optional
from metrics import MetricsTracker

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json


# ══════════════════════════════════════════════════════════════════════════════
# 0. Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Replay Buffers (MODIFIED OFFLINE)
# ══════════════════════════════════════════════════════════════════════════════

class OfflineBuffer:
    def __init__(self, path: str, device: torch.device):
        raw = np.load(path)

        self.states      = torch.tensor(raw["states"], dtype=torch.float32, device=device)
        self.actions     = torch.tensor(raw["actions"], dtype=torch.float32, device=device)  # ✅ already continuous
        self.rewards     = torch.tensor(raw["rewards"], dtype=torch.float32, device=device)
        self.next_states = torch.tensor(raw["next_states"], dtype=torch.float32, device=device)
        self.dones       = torch.tensor(raw["dones"], dtype=torch.float32, device=device)

        self.size   = len(self.states)
        self.device = device

        print(f"[OfflineBuffer] Loaded {self.size:,} transitions (continuous actions)")

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }


class OnlineBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device   = device
        self._ptr     = 0
        self._full    = False

        self.states      = torch.zeros((capacity, state_dim), device=device)
        self.actions     = torch.zeros((capacity, action_dim), device=device)
        self.rewards     = torch.zeros((capacity,), device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.dones       = torch.zeros((capacity,), device=device)

    @property
    def size(self):
        return self.capacity if self._full else self._ptr

    def add(self, s, a, r, s2, d):
        i = self._ptr
        self.states[i] = torch.tensor(s, device=self.device)
        self.actions[i] = torch.tensor(a, device=self.device)
        self.rewards[i] = r
        self.next_states[i] = torch.tensor(s2, device=self.device)
        self.dones[i] = d
        self._ptr = (self._ptr + 1) % self.capacity
        if self._ptr == 0:
            self._full = True

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Hy-Q Mixer (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

class HyQMixer:
    def __init__(self, offline_buf, online_buf,
                 beta_start=1.0, beta_end=0.25,
                 anneal_steps=50000, td_alpha=0.6):
        self.offline = offline_buf
        self.online  = online_buf
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_steps = anneal_steps
        self.td_alpha = td_alpha
        self._step = 0
        self._priorities = np.ones(len(self.offline.states))

    @property
    def beta(self):
        frac = min(self._step / self.anneal_steps, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def sample(self, batch_size):
        self._step += 1
        n_off = int(self.beta * batch_size)
        n_on  = batch_size - n_off

        batch_off = self.offline.sample(n_off) if n_off > 0 else None
        batch_on  = self.online.sample(n_on) if n_on > 0 else None

        batches = [b for b in [batch_off, batch_on] if b is not None]

        merged = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}
        return merged, None


# ══════════════════════════════════════════════════════════════════════════════
# 4. Diffusion (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class EpsilonNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.time_emb = SinusoidalTimeEmbedding(16)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 16, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, action_dim)
        )

    def forward(self, a, s, t):
        return self.net(torch.cat([a, s, self.time_emb(t)], dim=-1))


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
            # omega_s  = cfg.omega_ent * w_sel                             # (B,)

        # Q-weighted VLO loss  (Eq. 6)
        loss_q = self.diffusion.q_weighted_vlo_loss(self.eps_net, a_sel, s, w_sel)

        # # Entropy regularisation  (Eq. 10)
        # loss_e = self.diffusion.entropy_loss(
        #     self.eps_net, s, omega_s, cfg.Ne, cfg.force_mag
        # )
        return loss_q #+ loss_e

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

    def online_finetune(self, env):
        cfg = self.cfg
        print(f"{'='*60}")
        print(f"  Phase 2 — Online Finetuning ({cfg.online_steps:,} steps)")
        print(f"{'='*60}")

        state, _  = env.reset(seed=cfg.seed)
        ep_return = 0.0
        ep_count  = 0
        tracker = MetricsTracker("HyQVPO")

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
                # ✅ STEP-BASED EVALUATION (aligned with QVPO)
                if step % cfg.eval_interval_eps == 0 and step > 0:

                    returns = []
                    env_eval = env(seed=cfg.seed + 999)

                    for ep in range(cfg.eval_episodes):
                        s, _ = env_eval.reset(seed=cfg.seed + ep)
                        done = False
                        ep_ret = 0.0

                        while not done:
                            a = self.select_action(s)
                            s, r, term, trunc, _ = env_eval.step(a)
                            ep_ret += r
                            done = term or trunc

                        returns.append(ep_ret)

                    env_eval.close()

                    tracker.log_eval(
                        step=step,
                        returns=returns
                    )

                    avg = np.mean(returns)

                    print(f"  [online {step:7d}/{cfg.online_steps}]  "
                        f"avg_return={avg:6.1f}  β={self.mixer.beta:.3f}  "
                        f"online={self.online_buf.size}")


            if self.online_buf.size < cfg.batch_size:
                continue

            batch, off_idx = self.mixer.sample(cfg.batch_size)
            c_loss, p_loss = self._update_step(batch, off_idx)

            # For comparison
            tracker.log_step(
                step=step,
                policy_loss=p_loss,
                critic_loss=c_loss
            )

            self.log["critic_loss"].append(c_loss)
            self.log["policy_loss"].append(p_loss)
        tracker.save("Mujococo_HyQVPO_metrics.npz")

        print("  Online finetuning done.\n")

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, env, n_episodes: int = 10) -> float:
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
# 7. CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def build_config():
    p = argparse.ArgumentParser()

    p.add_argument("--env", type=str, default="Hopper-v4")  # ✅ NEW
    p.add_argument("--demo_path", default="dataset.npz")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--online_capacity", type=int, default=200000)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cfg.env)

    cfg.state_dim  = env.observation_space.shape[0]   # ✅ dynamic
    cfg.action_dim = env.action_space.shape[0]

    # trainer = DiffusionQLTrainer(cfg, device, env)

    # print(f"Running {cfg.env} | state_dim={cfg.state_dim} | action_dim={cfg.action_dim}")


if __name__ == "__main__":
    main()
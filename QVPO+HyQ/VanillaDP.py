"""
Vanilla Diffusion Policy — Continuous CartPole
==============================================
Baseline: pure behaviour cloning (BC) via DDPM on the continuous CartPole
environment.  No Q-learning, no critic, no Hy-Q, no advantage weighting.

What this is
------------
A diffusion model trained by standard denoising score matching (DSM) to
imitate the offline demonstration data.  At inference time it generates
actions by running the full DDPM reverse chain conditioned on the current
state.  That is all.

What is deliberately absent (and why)
--------------------------------------
  ✗  Q-network / critic          — vanilla BC needs no value signal
  ✗  TD updates / Bellman target — no RL objective
  ✗  qadv / advantage weights    — QVPO contribution, excluded
  ✗  K-efficient selection       — QVPO contribution, excluded
  ✗  Entropy regularisation      — QVPO contribution, excluded
  ✗  HyQMixer / online buffer    — Hy-Q contribution, excluded
  ✗  Online finetuning phase     — pure offline BC only
  ✗  Target networks             — nothing to soft-update

Training objective
------------------
  L(θ) = E_{a₀~D, ε~N(0,I), t} [ ||ε − ε_θ(√ᾱ_t·a₀ + √(1−ᾱ_t)·ε, s, t)||² ]

  a₀ is a demo action, s is the paired state.  The noise network ε_θ
  learns to denoise, making the model a generative model of p(a|s).

Environment
-----------
  ContinuousCartPoleEnv  —  identical physics to CartPole-v1 but with a
  scalar force f ∈ [-force_mag, +force_mag] replacing the bang-bang ±10 N.
  Demo actions (discrete 0/1) are remapped to {-force_mag, +force_mag} once
  at load time, placing them at valid extremes of the continuous action space.

Usage
-----
  python vanilla_diffusion.py --demo_path cartpole_demo_data.npz

  Key flags:
    --n_diffusion_steps   DDPM chain length (default 5)
    --train_steps         gradient steps on the offline data (default 20 000)
    --eval_episodes       episodes to run after training (default 20)
"""

import os
import math
import random
import argparse
from typing import Optional, Tuple
from metrics import MetricsTracker

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json


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
# 1.  Continuous CartPole environment  (shared with diffusion_ql.py)
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousCartPoleEnv:
    """
    CartPole with a native continuous scalar force action.

    Dynamics are identical to CartPole-v1 (Barto, Sutton & Anderson 1983)
    except the applied force is the raw scalar f ∈ [-force_mag, +force_mag]
    rather than a bang-bang ±force_mag signal.

    State  : [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    Action : scalar force  f ∈ [-force_mag, +force_mag]
    Reward : +1.0 every step the pole remains upright
    Done   : |pole| > 12 °  OR  |cart| > 2.4 m  OR  steps ≥ max_steps
    """

    GRAVITY      = 9.8
    MASSCART     = 1.0
    MASSPOLE     = 0.1
    TOTAL_MASS   = MASSCART + MASSPOLE
    HALF_LEN     = 0.5
    POLEMASS_LEN = MASSPOLE * HALF_LEN
    TAU          = 0.02

    THETA_THRESHOLD = 12 * 2 * math.pi / 360   # radians
    X_THRESHOLD     = 2.4                       # metres

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
        f = float(np.clip(action, -self.force_mag, self.force_mag))
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
# 2.  Demo dataset buffer  (offline only, no online ring-buffer)
# ══════════════════════════════════════════════════════════════════════════════

class DemoBuffer:
    """
    Loads cartpole_demo_data.npz and converts discrete actions {0,1} to
    continuous forces {-force_mag, +force_mag}.

    Only (state, action) pairs are needed for BC — rewards, next_states,
    and dones are not used anywhere in vanilla diffusion policy training.
    They are loaded anyway so the buffer could be reused in ablations.
    """

    def __init__(self, path: str, device: torch.device, force_mag: float = 10.0):
        raw        = np.load(path)
        states     = torch.tensor(raw["states"],      dtype=torch.float32)
        actions_d  = torch.tensor(raw["actions"],     dtype=torch.float32)  # 0 or 1
        rewards    = torch.tensor(raw["rewards"],     dtype=torch.float32)
        next_states= torch.tensor(raw["next_states"], dtype=torch.float32)

        # discrete {0,1} → continuous {-force_mag, +force_mag}, shape (N, 1)
        actions_c = ((actions_d * 2.0 - 1.0) * force_mag).unsqueeze(-1)

        self.states      = states.to(device)
        self.actions     = actions_c.to(device)       # (N, 1)  continuous
        self.rewards     = rewards.to(device)
        self.next_states = next_states.to(device)
        self.size        = len(states)
        self.device      = device

        print(f"[DemoBuffer] {self.size:,} transitions | "
              f"actions: 0 → {-force_mag:.1f} N,  1 → {+force_mag:.1f} N")

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (states, actions) — the only two tensors BC needs."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.states[idx], self.actions[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DDPM diffusion model  (shared math with diffusion_ql.py)
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
    Noise prediction network  ε_θ(ã_t, s, t) → ε̂.

    Identical architecture to diffusion_ql.py — the only thing that changes
    between vanilla BC and QVPO is how this network is *trained*, not its
    structure.

    Input  : cat(ã_t, s, t_emb)
    Output : ε̂  ∈  R^{action_dim}
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, time_emb_dim: int = 16, n_steps: int = 5):
        super().__init__()
        self.action_dim = action_dim
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
        return self.net(torch.cat([a_t, state, t_emb], dim=-1))


class GaussianDiffusion:
    """
    DDPM schedule + forward/reverse process for continuous actions.

    Forward  q(ã_t | a₀) = N(√ᾱ_t · a₀,  (1−ᾱ_t) · I)
    Reverse  p_θ(ã_{t-1} | ã_t, s)  via  EpsilonNet

    Training loss (vanilla BC — the *only* loss used here):

        L(θ) = E_{a₀~D, ε~N(0,I), t~U[1,T]} [ ||ε − ε_θ(ã_t, s, t)||² ]

    where  ã_t = √ᾱ_t · a₀ + √(1−ᾱ_t) · ε.
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

    def to(self, device: torch.device) -> "GaussianDiffusion":
        for attr in ["betas", "alpha_bar", "alpha_bar_prev",
                     "sqrt_ab", "sqrt_1mab", "posterior_var"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ── Forward process q(ã_t | a₀) ───────────────────────────────────────

    def q_sample(self, a0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (ã_t, ε)."""
        if noise is None:
            noise = torch.randn_like(a0)
        s_ab   = self.sqrt_ab[t - 1].view(-1, 1)
        s_1mab = self.sqrt_1mab[t - 1].view(-1, 1)
        return s_ab * a0 + s_1mab * noise, noise

    # ── Vanilla BC training loss ───────────────────────────────────────────

    def bc_loss(self, eps_net: EpsilonNet,
                a0: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Plain DDPM denoising MSE loss — the one and only training objective.

        L(θ) = E_{ε, t} [ ||ε − ε_θ(√ᾱ_t·a₀ + √(1−ᾱ_t)·ε, s, t)||² ]
        """
        B          = a0.shape[0]
        t          = torch.randint(1, self.T + 1, (B,), device=a0.device)
        a_t, noise = self.q_sample(a0, t)
        eps_hat    = eps_net(a_t, state, t)
        return ((noise - eps_hat) ** 2).mean()

    # ── Single reverse step ────────────────────────────────────────────────

    @torch.no_grad()
    def _p_sample_step(self, eps_net: EpsilonNet, a_t: torch.Tensor,
                       state: torch.Tensor, t_val: int) -> torch.Tensor:
        B       = a_t.shape[0]
        t       = torch.full((B,), t_val, dtype=torch.long, device=a_t.device)
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

    # ── Full reverse chain: action generation ─────────────────────────────

    @torch.no_grad()
    def p_sample(self, eps_net: EpsilonNet,
                 state: torch.Tensor) -> torch.Tensor:
        """
        Generate one action per state by running the full reverse chain.

        No K-efficient selection, no Q-guidance — a single sample per state.

        Returns (B, action_dim).
        """
        a_t = torch.randn(state.shape[0], eps_net.action_dim, device=state.device)
        for t_val in reversed(range(1, self.T + 1)):
            a_t = self._p_sample_step(eps_net, a_t, state, t_val)
        return a_t


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Vanilla Diffusion Policy trainer
# ══════════════════════════════════════════════════════════════════════════════

class VanillaDiffusionPolicy:
    """
    Trains a DDPM-based behaviour cloning policy on offline demo data.

    One network (EpsilonNet), one loss (bc_loss), one optimiser.
    No critics, no replay buffers beyond the demo dataset, no RL.

    Training loop
    -------------
      for each gradient step:
          (s, a₀) ~ DemoBuffer                     # sample a mini-batch
          t       ~ Uniform{1, ..., T}              # random diffusion step
          ε       ~ N(0, I)                         # random noise
          ã_t      = √ᾱ_t·a₀ + √(1−ᾱ_t)·ε         # forward process
          ε̂       = ε_θ(ã_t, s, t)                # noise prediction
          L        = ||ε − ε̂||²                    # MSE
          update θ via Adam

    Inference
    ---------
      a ~ p_θ(·|s)  by running the full reverse chain from Gaussian noise.
      One sample per state — no argmax, no candidate selection.
    """

    def __init__(self, cfg: argparse.Namespace, device: torch.device):
        self.cfg    = cfg
        self.device = device

        # ── Demo data ─────────────────────────────────────────────────────────
        self.demo = DemoBuffer(cfg.demo_path, device, cfg.force_mag)

        # ── Diffusion schedule ────────────────────────────────────────────────
        self.diffusion = GaussianDiffusion(
            n_steps=cfg.n_diffusion_steps,
            beta_min=cfg.beta_min,
            beta_max=cfg.beta_max,
        ).to(device)

        # ── Noise network ─────────────────────────────────────────────────────
        self.eps_net = EpsilonNet(
            state_dim    = cfg.state_dim,
            action_dim   = cfg.action_dim,
            hidden_dim   = cfg.hidden_dim,
            time_emb_dim = cfg.time_emb_dim,
            n_steps      = cfg.n_diffusion_steps,
        ).to(device)

        # ── Single optimiser ──────────────────────────────────────────────────
        self.opt = optim.Adam(self.eps_net.parameters(), lr=cfg.lr)

        self.log = {"bc_loss": [], "eval_return": []}

    # ── Training ───────────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  Vanilla Diffusion Policy — BC Training ({cfg.train_steps:,} steps)")
        print(f"  T={cfg.n_diffusion_steps}  hidden={cfg.hidden_dim}  "
              f"batch={cfg.batch_size}  lr={cfg.lr}")
        print(f"{'='*60}\n")
        tracker = MetricsTracker("VanillaDiffusion")

        for step in range(1, cfg.train_steps + 1):
            states, actions = self.demo.sample(cfg.batch_size)

            loss = self.diffusion.bc_loss(self.eps_net, actions, states)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.eps_net.parameters(), 1.0)
            self.opt.step()

            self.log["bc_loss"].append(loss.item())

            # For comparison
            tracker.log_step(
                step=step,
                policy_loss=loss.item(),
                critic_loss=None   # no critic in vanilla
            )


            if step % cfg.log_interval == 0:
                avg_loss = sum(self.log["bc_loss"][-cfg.log_interval:]) / cfg.log_interval
                print(f"  step {step:6d}/{cfg.train_steps}  bc_loss={avg_loss:.6f}")
            
            # if step % cfg.eval_interval == 0:
            #     returns = self.evaluate(n_episodes=10)

                tracker.log_eval(
                    step=step,
                    returns=self.evaluate(n_episodes=10)
                )
            tracker.save("vanilla_diffusion_metrics.npz") 
        print("\n  Training complete.\n")

    # ── Action selection — single DDPM sample, no selection tricks ────────

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Run the full reverse chain for one state; return (action_dim,) array."""
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.diffusion.p_sample(self.eps_net, s)   # (1, action_dim)
        return a.squeeze(0).cpu().numpy()

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, env: ContinuousCartPoleEnv, n_episodes: int = 20) -> float:
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
            print(f"    ep {ep+1:2d}:  return = {ep_ret:.0f}")
        mean_ret = float(np.mean(returns))
        std_ret  = float(np.std(returns))
        print(f"\n  → mean = {mean_ret:.1f}   std = {std_ret:.1f}   "
              f"max = {max(returns):.0f}   min = {min(returns):.0f}")
        self.log["eval_return"].append(mean_ret)
        return mean_ret

    # ── Checkpoint ─────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"eps_net": self.eps_net.state_dict()}, path)
        print(f"  Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.eps_net.load_state_dict(ckpt["eps_net"])
        print(f"  Checkpoint loaded ← {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Configuration & entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Vanilla Diffusion Policy (BC) — Continuous CartPole"
    )

    # Environment
    p.add_argument("--state_dim",  type=int,   default=4)
    p.add_argument("--action_dim", type=int,   default=1)
    p.add_argument("--force_mag",  type=float, default=10.0,
                   help="Max cart force ±F (N).  Demos remapped to {-F, +F}.")
    p.add_argument("--max_steps",  type=int,   default=500)
    p.add_argument("--seed",       type=int,   default=42)

    # Demo data
    p.add_argument("--demo_path",  default="cartpole_demo_data.npz")

    # Diffusion model
    p.add_argument("--n_diffusion_steps", type=int,   default=5,
                   help="DDPM chain length T")
    p.add_argument("--beta_min",          type=float, default=0.1)
    p.add_argument("--beta_max",          type=float, default=0.5)
    p.add_argument("--hidden_dim",        type=int,   default=256)
    p.add_argument("--time_emb_dim",      type=int,   default=16)

    # Training
    p.add_argument("--train_steps",  type=int,   default=100_000,
                   help="Gradient steps on demo data")
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--log_interval", type=int,   default=1_000)

    # Evaluation
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--save_path",     default="checkpoints/vanilla_diffusion.pt")

    return p.parse_args()


def main():
    cfg    = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    set_seed(cfg.seed)

    env    = ContinuousCartPoleEnv(force_mag=cfg.force_mag,
                                   max_steps=cfg.max_steps, seed=cfg.seed)
    policy = VanillaDiffusionPolicy(cfg, device)

    policy.train()
    policy.save(cfg.save_path)

    print(f"\n  [Eval] {cfg.eval_episodes} episodes on ContinuousCartPole:")
    policy.evaluate(env, n_episodes=cfg.eval_episodes)

    env.close()


if __name__ == "__main__":
    main()
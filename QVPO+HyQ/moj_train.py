import os
import math
import random
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym


# =============================================================================
# 0. Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. Replay Buffer
# =============================================================================

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, size=int(1e6), device="cpu"):
        self.device = device
        self.ptr = 0
        self.size = 0
        self.max_size = size

        self.state = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((size, 1), dtype=torch.float32, device=device)

    def add(self, s, a, r, s2, d):
        self.state[self.ptr] = torch.tensor(s, device=self.device)
        self.action[self.ptr] = torch.tensor(a, device=self.device)
        self.reward[self.ptr] = torch.tensor([r], device=self.device)
        self.next_state[self.ptr] = torch.tensor(s2, device=self.device)
        self.done[self.ptr] = torch.tensor([d], device=self.device)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx],
        )


# =============================================================================
# 2. Diffusion Model
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class EpsilonNet(nn.Module):
    def __init__(self, state_dim, action_dim, time_dim=32):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + time_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, a_t, state, t):
        t_embed = self.time_embed(t)
        x = torch.cat([a_t, state, t_embed], dim=-1)
        return self.net(x)


class GaussianDiffusion:
    def __init__(self, action_dim, T=50, device="cpu"):
        self.T = T
        self.device = device

        beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.action_dim = action_dim

    def sample(self, model, state):
        B = state.shape[0]
        x = torch.randn(B, self.action_dim).to(self.device)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.float32)
            eps = model(x, state, t_tensor)

            a = self.alpha[t]
            abar = self.alpha_bar[t]

            x = (x - (1 - a) / torch.sqrt(1 - abar) * eps) / torch.sqrt(a)

            if t > 0:
                x += torch.randn_like(x) * 0.01

        return x


# =============================================================================
# 3. Q Network
# =============================================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# =============================================================================
# 4. Trainer
# =============================================================================

class Trainer:
    def __init__(self, env, cfg, device):
        self.env = env
        self.device = device

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.action_low = torch.tensor(env.action_space.low, device=device)
        self.action_high = torch.tensor(env.action_space.high, device=device)

        self.policy = EpsilonNet(self.state_dim, self.action_dim).to(device)
        self.diffusion = GaussianDiffusion(self.action_dim, device=device)

        self.q1 = QNetwork(self.state_dim, self.action_dim).to(device)
        self.q2 = QNetwork(self.state_dim, self.action_dim).to(device)

        self.q1_target = QNetwork(self.state_dim, self.action_dim).to(device)
        self.q2_target = QNetwork(self.state_dim, self.action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4)

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim, device=device)

        self.gamma = 0.99

    def scale_action(self, a):
        a = torch.tanh(a)
        return self.action_low + (a + 1.0) * 0.5 * (self.action_high - self.action_low)

    def train_step(self, batch_size):
        s, a, r, s2, d = self.buffer.sample(batch_size)

        with torch.no_grad():
            next_a = self.diffusion.sample(self.policy, s2)
            next_a = self.scale_action(next_a)

            q_target = torch.min(
                self.q1_target(s2, next_a),
                self.q2_target(s2, next_a),
            )
            target = r + (1 - d) * self.gamma * q_target

        q1_loss = F.mse_loss(self.q1(s, a), target)
        q2_loss = F.mse_loss(self.q2(s, a), target)

        self.q_optim.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optim.step()

    def rollout(self, steps=1000):
        state, _ = self.env.reset()

        total_reward = 0

        for _ in range(steps):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action = self.diffusion.sample(self.policy, state_t)
                action = self.scale_action(action).cpu().numpy()[0]

            next_state, reward, done, trunc, _ = self.env.step(action)

            self.buffer.add(state, action, reward, next_state, done or trunc)

            state = next_state
            total_reward += reward

            if done or trunc:
                state, _ = self.env.reset()

        return total_reward


# =============================================================================
# 5. Main
# =============================================================================

def build_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v3")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    cfg = build_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)

    env = gym.make(cfg.env)

    trainer = Trainer(env, cfg, device)

    for step in range(cfg.steps):
        reward = trainer.rollout(1000)

        if trainer.buffer.size > cfg.batch:
            for _ in range(50):
                trainer.train_step(cfg.batch)

        if step % 10 == 0:
            print(f"[Step {step}] Reward: {reward:.2f}")


if __name__ == "__main__":
    main()
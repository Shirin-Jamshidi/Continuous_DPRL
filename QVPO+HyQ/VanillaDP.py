import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ==============================
# 1. Continuous CartPole Env
# ==============================
class ContinuousCartPoleEnv:
    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    TOTAL_MASS = MASSCART + MASSPOLE
    HALF_LEN = 0.5
    POLEMASS_LEN = MASSPOLE * HALF_LEN
    TAU = 0.02

    THETA_THRESHOLD = 12 * 2 * math.pi / 360
    X_THRESHOLD = 2.4

    def __init__(self, force_mag=10.0, max_steps=500):
        self.force_mag = force_mag
        self.max_steps = max_steps
        self.rng = np.random.RandomState(42)
        self.state = None
        self.steps = 0

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        f = float(np.clip(action, -self.force_mag, self.force_mag))
        x, x_dot, theta, theta_dot = self.state

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        tmp = (f + self.POLEMASS_LEN * theta_dot**2 * sin_t) / self.TOTAL_MASS
        theta_acc = (self.GRAVITY * sin_t - cos_t * tmp) / (
            self.HALF_LEN * (4.0 / 3.0 - self.MASSPOLE * cos_t**2 / self.TOTAL_MASS)
        )
        x_acc = tmp - self.POLEMASS_LEN * theta_acc * cos_t / self.TOTAL_MASS

        x += self.TAU * x_dot
        x_dot += self.TAU * x_acc
        theta += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        done = (
            abs(x) > self.X_THRESHOLD
            or abs(theta) > self.THETA_THRESHOLD
            or self.steps >= self.max_steps
        )

        return self.state, 1.0, done, False, {}


# ==============================
# 2. Offline Buffer
# ==============================
class OfflineBuffer:
    def __init__(self, path, device):
        data = np.load(path)

        self.states = torch.tensor(data["states"], dtype=torch.float32, device=device)
        self.actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)

        if self.actions.dim() == 1:
            self.actions = self.actions.unsqueeze(-1)

        self.size = len(self.states)
        self.device = device

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
        }


# ==============================
# 3. Diffusion Model
# ==============================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class EpsilonNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, time_dim=16):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + time_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, a_t, s, t):
        t_emb = self.time_emb(t)
        x = torch.cat([a_t, s, t_emb], dim=-1)
        return self.net(x)


class GaussianDiffusion:
    def __init__(self, T=5, beta_min=0.1, beta_max=0.5):
        self.T = T
        betas = torch.linspace(beta_min, beta_max, T)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(self, a0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(a0)

        ab = self.alpha_bar[t - 1].view(-1, 1)
        return ab.sqrt() * a0 + (1 - ab).sqrt() * noise, noise

    @torch.no_grad()
    def sample(self, model, state):
        a = torch.randn(state.shape[0], 1, device=state.device)
        for t in reversed(range(1, self.T + 1)):
            t_tensor = torch.full((state.shape[0],), t, device=state.device)
            eps = model(a, state, t_tensor)
            ab = self.alpha_bar[t - 1]
            a = (a - (1 - ab).sqrt() * eps) / ab.sqrt()
        return a


# ==============================
# 4. Trainer
# ==============================
class Trainer:
    def __init__(self, cfg, device):
        self.device = device

        self.buffer = OfflineBuffer(cfg.demo_path, device)

        self.model = EpsilonNet(cfg.state_dim, cfg.action_dim).to(device)
        self.diffusion = GaussianDiffusion(cfg.T).to(device)

        self.opt = optim.Adam(self.model.parameters(), lr=3e-4)

    def train(self, cfg):
        for step in range(cfg.steps):
            batch = self.buffer.sample(cfg.batch)

            s = batch["states"]
            a = batch["actions"]

            B = a.shape[0]
            t = torch.randint(1, self.diffusion.T + 1, (B,), device=self.device)

            a_t, noise = self.diffusion.q_sample(a, t)
            pred = self.model(a_t, s, t)

            loss = F.mse_loss(pred, noise)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if step % 1000 == 0:
                print(f"step {step} loss {loss.item():.4f}")

    @torch.no_grad()
    def act(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.diffusion.sample(self.model, s)
        return a[0].cpu().numpy()

    def evaluate(self, env, episodes=10):
        for ep in range(episodes):
            s, _ = env.reset()
            total = 0

            while True:
                a = self.act(s)
                s, r, done, _, _ = env.step(a)
                total += r
                if done:
                    break

            print(f"Episode {ep+1}: {total}")


# ==============================
# 5. Main
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", default="cartpole_demo_data.npz")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--state_dim", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=1)
    parser.add_argument("--T", type=int, default=5)

    cfg = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(cfg, device)
    env = ContinuousCartPoleEnv()

    print("Training...")
    trainer.train(cfg)

    print("Evaluating...")
    trainer.evaluate(env)


if __name__ == "__main__":
    main()
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ============================================================
# ✅ Hyperparameters
# ============================================================
NUM_EPISODES = 3000
MAX_STEPS = 500
GAMMA = 0.99
LR = 1e-3

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 30000

NUM_DEMO_EPISODES = 2000
FORCE_MAG = 10.0   # MUST match train.py ContinuousCartPoleEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ✅ Environment (standard Gym)
# ============================================================
env = gym.make("CartPole-v1")

# ============================================================
# ✅ Q-Network (Continuous state → discrete actions)
# ============================================================
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # left / right
        )

    def forward(self, x):
        return self.net(x)

q_net = QNet().to(DEVICE)
optimizer = optim.Adam(q_net.parameters(), lr=LR)

# ============================================================
# ✅ Epsilon schedule
# ============================================================
def epsilon_by_step(step):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-step / EPS_DECAY)

# ============================================================
# ✅ TRAIN Q-NETWORK
# ============================================================
print("Training Q-network...")

step_count = 0
reward_history = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        state = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        eps = epsilon_by_step(step_count)
        step_count += 1

        # ✅ Epsilon-greedy
        if random.random() < eps:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_vals = q_net(state)
                action = torch.argmax(q_vals).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

        # ✅ Q-learning update
        q_values = q_net(state)
        q_value = q_values[action]

        with torch.no_grad():
            next_q = q_net(next_state)
            target = reward + (0 if done else GAMMA * torch.max(next_q))

        loss = (q_value - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs
        total_reward += reward

        if done:
            break

    reward_history.append(total_reward)

    if episode % 100 == 0:
        print(f"[TRAIN] Episode {episode}, Reward: {total_reward:.1f}, Eps: {eps:.3f}")

# ============================================================
# ✅ DEMONSTRATION DATA COLLECTION
# ============================================================
print("\nCollecting demonstration dataset...")

states = []
actions = []
rewards = []
next_states = []
dones = []

for episode in range(NUM_DEMO_EPISODES):
    obs, _ = env.reset()

    for step in range(MAX_STEPS):
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        # ✅ Greedy policy
        with torch.no_grad():
            q_vals = q_net(state_tensor)
            discrete_action = torch.argmax(q_vals).item()

        # ✅ Convert to continuous force
        action_cont = -FORCE_MAG if discrete_action == 0 else FORCE_MAG

        # ✅ small noise (helps diffusion a LOT)
        noise = np.random.normal(0, 1.0)
        action_cont = np.clip(action_cont + noise, -FORCE_MAG, FORCE_MAG)

        next_obs, reward, terminated, truncated, _ = env.step(discrete_action)
        done = terminated or truncated

        # ✅ Store transition
        states.append(obs)
        actions.append(action_cont)  # shape (1,)
        rewards.append(reward)
        next_states.append(next_obs)
        dones.append(done)

        obs = next_obs

        if done:
            break

    if episode % 50 == 0:
        print(f"[DEMO] Episode {episode}")

# ============================================================
# ✅ SAVE DATASET
# ============================================================
states = np.array(states, dtype=np.float32)
actions = np.array(actions, dtype=np.float32).reshape(-1, 1)  
rewards = np.array(rewards, dtype=np.float32)
next_states = np.array(next_states, dtype=np.float32)
dones = np.array(dones, dtype=np.bool_)

np.savez(
    "cartpole_demo_data.npz",
    states=states,
    actions=actions,
    rewards=rewards,
    next_states=next_states,
    dones=dones,
)

print("\n✅ Dataset saved: cartpole_demo_data.npz")
print("Size:", len(states))

# ============================================================
# ✅ PLOT TRAINING
# ============================================================
plt.figure()
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-Network CartPole Training")
plt.savefig("offline_training_curve.png")
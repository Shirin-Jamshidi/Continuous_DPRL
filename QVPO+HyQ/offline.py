# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import matplotlib.pyplot as plt

# # ============================================================
# # ✅ Hyperparameters
# # ============================================================
# NUM_EPISODES = 3000
# MAX_STEPS = 500
# GAMMA = 0.99
# LR = 1e-3

# EPS_START = 1.0
# EPS_END = 0.01
# EPS_DECAY = 30000

# NUM_DEMO_EPISODES = 2000
# FORCE_MAG = 10.0   # MUST match train.py ContinuousCartPoleEnv

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ============================================================
# # ✅ Environment (standard Gym)
# # ============================================================
# env = gym.make("CartPole-v1")

# # ============================================================
# # ✅ Q-Network (Continuous state → discrete actions)
# # ============================================================
# class QNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(4, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2)  # left / right
#         )

#     def forward(self, x):
#         return self.net(x)

# q_net = QNet().to(DEVICE)
# optimizer = optim.Adam(q_net.parameters(), lr=LR)

# # ============================================================
# # ✅ Epsilon schedule
# # ============================================================
# def epsilon_by_step(step):
#     return EPS_END + (EPS_START - EPS_END) * np.exp(-step / EPS_DECAY)

# # ============================================================
# # ✅ TRAIN Q-NETWORK
# # ============================================================
# print("Training Q-network...")

# step_count = 0
# reward_history = []

# for episode in range(NUM_EPISODES):
#     obs, _ = env.reset()
#     total_reward = 0

#     for step in range(MAX_STEPS):
#         state = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

#         eps = epsilon_by_step(step_count)
#         step_count += 1

#         # ✅ Epsilon-greedy
#         if random.random() < eps:
#             action = random.randint(0, 1)
#         else:
#             with torch.no_grad():
#                 q_vals = q_net(state)
#                 action = torch.argmax(q_vals).item()

#         next_obs, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated

#         next_state = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

#         # ✅ Q-learning update
#         q_values = q_net(state)
#         q_value = q_values[action]

#         with torch.no_grad():
#             next_q = q_net(next_state)
#             target = reward + (0 if done else GAMMA * torch.max(next_q))

#         loss = (q_value - target) ** 2

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         obs = next_obs
#         total_reward += reward

#         if done:
#             break

#     reward_history.append(total_reward)

#     if episode % 100 == 0:
#         print(f"[TRAIN] Episode {episode}, Reward: {total_reward:.1f}, Eps: {eps:.3f}")

# # ============================================================
# # ✅ DEMONSTRATION DATA COLLECTION
# # ============================================================
# print("\nCollecting demonstration dataset...")

# states = []
# actions = []
# rewards = []
# next_states = []
# dones = []

# for episode in range(NUM_DEMO_EPISODES):
#     obs, _ = env.reset()

#     for step in range(MAX_STEPS):
#         state_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

#         # ✅ Greedy policy
#         with torch.no_grad():
#             q_vals = q_net(state_tensor)
#             discrete_action = torch.argmax(q_vals).item()

#         # ✅ Convert to continuous force
#         action_cont = -FORCE_MAG if discrete_action == 0 else FORCE_MAG

#         # ✅ small noise (helps diffusion a LOT)
#         noise = np.random.normal(0, 1.0)
#         action_cont = np.clip(action_cont + noise, -FORCE_MAG, FORCE_MAG)

#         next_obs, reward, terminated, truncated, _ = env.step(discrete_action)
#         done = terminated or truncated

#         # ✅ Store transition
#         states.append(obs)
#         actions.append(action_cont)  # shape (1,)
#         rewards.append(reward)
#         next_states.append(next_obs)
#         dones.append(done)

#         obs = next_obs

#         if done:
#             break

#     if episode % 50 == 0:
#         print(f"[DEMO] Episode {episode}")

# # ============================================================
# # ✅ SAVE DATASET
# # ============================================================
# states = np.array(states, dtype=np.float32)
# actions = np.array(actions, dtype=np.float32).reshape(-1, 1)  
# rewards = np.array(rewards, dtype=np.float32)
# next_states = np.array(next_states, dtype=np.float32)
# dones = np.array(dones, dtype=np.bool_)

# np.savez(
#     "cartpole_demo_data.npz",
#     states=states,
#     actions=actions,
#     rewards=rewards,
#     next_states=next_states,
#     dones=dones,
# )

# print("\n✅ Dataset saved: cartpole_demo_data.npz")
# print("Size:", len(states))

# # ============================================================
# # ✅ PLOT TRAINING
# # ============================================================
# plt.figure()
# plt.plot(reward_history)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title("Q-Network CartPole Training")
# plt.savefig("offline_training_curve.png")



import gymnasium as gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# --- Hyperparameters ---
NUM_BUCKETS = (10, 10, 20, 20)
NUM_ACTIONS = 31
NUM_EPISODES = 30000
MAX_STEPS = 500

MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

DECAY_FACTOR = 25

# --- Initialize env ---
env = gym.make("CartPole-v1")

# State bounds
state_bounds = [
    (-2.4, 2.4),          # cart position
    (-3.0, 3.0),          # cart velocity (clip)
    (-0.2095, 0.2095),    # pole angle (~12 degrees)
    (-3.5, 3.5)           # pole angular velocity (clip)
]
# Q-table
q_table = np.zeros(NUM_BUCKETS + (env.action_space.n,))

# --- Helper functions ---
def discretize(obs):
    ratios = [
        (obs[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0])
        for i in range(len(obs))
    ]
    new_obs = [
        int(round((NUM_BUCKETS[i] - 1) * ratios[i]))
        for i in range(len(obs))
    ]
    new_obs = [
        min(NUM_BUCKETS[i] - 1, max(0, new_obs[i]))
        for i in range(len(obs))
    ]
    return tuple(new_obs)

def choose_action(state, explore_rate):
    if random.random() < explore_rate:
        return env.action_space.sample()
    return np.argmax(q_table[state])

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1.0, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))

# =======================
# ✅ TRAINING PHASE
# =======================
rewards = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    state = discretize(obs)

    explore_rate = get_explore_rate(episode)
    learning_rate = get_learning_rate(episode)

    total_reward = 0

    for step in range(MAX_STEPS):
        action = choose_action(state, explore_rate)
        obs, reward, terminated, truncated, _ = env.step(action)

        new_state = discretize(obs)

        # Q update
        q_table[state + (action,)] += learning_rate * (
            reward + 0.99 * np.max(q_table[new_state]) - q_table[state + (action,)]
        )

        state = new_state
        total_reward += reward

        if terminated or truncated:
            break

    rewards.append(total_reward)

    if episode % 100 == 0:
        print(f"[TRAIN] Episode {episode}, Reward: {total_reward}")

# =======================
# ✅ DEMONSTRATION COLLECTION
# =======================

print("\nCollecting demonstration data...")

states = []
actions = []
rewards_demo = []
next_states = []
dones = []

NUM_DEMO_EPISODES = 2000

for episode in range(NUM_DEMO_EPISODES):
    obs, _ = env.reset()
    state = discretize(obs)

    for step in range(MAX_STEPS):

        # ✅ NO exploration → greedy policy
        action = np.argmax(q_table[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        new_state = discretize(next_obs)
    
        # ✅ store raw (continuous) observations
        states.append(obs)
        actions.append(action)
        rewards_demo.append(reward)
        next_states.append(next_obs)
        dones.append(terminated or truncated)

        obs = next_obs
        state = new_state

        if terminated or truncated:
            break

    if episode % 50 == 0:
        print(f"[DEMO] Episode {episode}")

# =======================
# ✅ SAVE DATASET
# =======================

states = np.array(states)
actions = np.array(actions)
rewards_demo = np.array(rewards_demo)
next_states = np.array(next_states)
dones = np.array(dones)

np.savez("cartpole_demo_data.npz",
         states=states,
         actions=actions,
         rewards=rewards_demo,
         next_states=next_states,
         dones=dones
         )

print("\nSaved dataset as cartpole_demo_data.npz")
print("Dataset size:", len(states))

# =======================
# ✅ Plot training result
# =======================
plt.figure()
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-learning CartPole")
plt.savefig("training_plot.png")
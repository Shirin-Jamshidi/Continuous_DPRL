import gymnasium as gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# --- Hyperparameters ---
NUM_BUCKETS = (10, 10, 20, 20)
NUM_ACTIONS = 31
NUM_EPISODES = 3000
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
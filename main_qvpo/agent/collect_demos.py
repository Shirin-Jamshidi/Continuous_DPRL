import argparse
import gym
import numpy as np
import torch
from stable_baselines3 import SAC

def train_agent(env_name):
    env = gym.make(env_name)
    # Create the agent
    agent = SAC(
        "MlpPolicy",
        env,
        verbose=1,
    )
    # Train the agent
    print(">>> Starting training...")
    agent.learn(100)
    print(">>> Training complete!")
    return agent  # MUST return the trained agent object

def collect(agent,
            env_name,
            save_path,
            episodes=10):

    env = gym.make(env_name)
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    print(f">>> Collecting {episodes} demonstration episodes...")
    for ep in range(episodes):
        # Handle gym return structure variation safely
        reset_output = env.reset()
        s = reset_output[0] if isinstance(reset_output, tuple) else reset_output
        done = False

        while not done:
            # Use the passed agent object to predict actions
            a, _ = agent.predict(
                s,
                deterministic=True
            )

            # Unpack step returns
            step_return = env.step(a)
            if len(step_return) == 5:
                ns, r, term, trunc, _ = step_return
                d = term or trunc
            else:
                ns, r, d, _ = step_return

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

            s = ns
            done = d

    np.savez_compressed(
        save_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32)
    )

    print(f">>> Successfully collected {len(states)} transitions to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env")
    parser.add_argument("--output")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    
    # 1. Train and get back the trained agent object
    trained_agent = train_agent(args.env)
    
    # 2. Pass that trained agent into collection
    collect(
        trained_agent,
        args.env,
        args.output,
        args.episodes
    )

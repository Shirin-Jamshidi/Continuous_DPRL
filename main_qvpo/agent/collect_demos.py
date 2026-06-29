import argparse
import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC


def collect(model_path,
            env_name,
            save_path,
            episodes=100):

    env = gym.make(env_name)

    model = SAC.load(model_path)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for ep in range(episodes):

        s, _ = env.reset()

        done = False

        while not done:

            a, _ = model.predict(
                s,
                deterministic=True
            )

            ns, r, term, trunc, _ = env.step(a)

            d = term or trunc

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

    print(f"Collected {len(states)} transitions")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model")
    parser.add_argument("--env")
    parser.add_argument("--output")
    parser.add_argument("--episodes", type=int, default=100)

    args = parser.parse_args()

    collect(
        args.model,
        args.env,
        args.output,
        args.episodes
    )
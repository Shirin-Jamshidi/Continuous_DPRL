import gymnasium as gym
import numpy as np
import argparse


def collect_random(env_name, num_steps, save_path):
    env = gym.make(env_name)

    s_list, a_list, r_list, s2_list, d_list = [], [], [], [], []

    s, _ = env.reset()

    for step in range(num_steps):
        a = env.action_space.sample()

        s2, r, done, trunc, _ = env.step(a)

        s_list.append(s)
        a_list.append(a)
        r_list.append([r])
        s2_list.append(s2)
        d_list.append([float(done or trunc)])

        s = s2

        if done or trunc:
            s, _ = env.reset()

        if step % 10000 == 0:
            print(f"Collected {step} steps")

    np.savez(
        save_path,
        s=np.array(s_list),
        a=np.array(a_list),
        r=np.array(r_list),
        s2=np.array(s2_list),
        d=np.array(d_list),
    )

    print(f"Saved dataset to {save_path}")


# =============================================================================
# OPTIONAL: better dataset using trained policy
# =============================================================================

def collect_policy(env_name, num_steps, policy, save_path):
    env = gym.make(env_name)

    s_list, a_list, r_list, s2_list, d_list = [], [], [], [], []

    s, _ = env.reset()

    for step in range(num_steps):
        a = policy(s)   # user-defined policy

        s2, r, done, trunc, _ = env.step(a)

        s_list.append(s)
        a_list.append(a)
        r_list.append([r])
        s2_list.append(s2)
        d_list.append([float(done or trunc)])

        s = s2

        if done or trunc:
            s, _ = env.reset()

    np.savez(save_path, s=s_list, a=a_list, r=r_list, s2=s2_list, d=d_list)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v4")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--save", type=str, default="hopper_dataset.npz")
    args = parser.parse_args()

    collect_random(args.env, args.steps, args.save)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import numpy as np
from envs.mini_clash import MiniClashEnv

def run_random(env, episodes=100):
    rewards = []
    for ep in range(episodes):
        obs, info = env.reset()
        total = 0.0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    print(f"[Random]  Avg reward over {episodes} episodes: {np.mean(rewards):.2f}")
    return rewards

def run_scripted(env, episodes=100):
    rewards = []
    for ep in range(episodes):
        obs, info = env.reset()
        total = 0.0
        while True:
            # simple scripted policy: spawn unit-type B (action=2) if enough elixir, else pass (0)
            your_elixir = obs[2]
            action = 2 if your_elixir >= 4 else 0
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    print(f"[Scripted] Avg reward over {episodes} episodes: {np.mean(rewards):.2f}")
    return rewards

def main():
    env = MiniClashEnv()
    print("\n--- Testing random agent ---")
    run_random(env, episodes=100)
    print("\n--- Testing simple scripted agent ---")
    run_scripted(env, episodes=100)

if __name__ == "__main__":
    main()

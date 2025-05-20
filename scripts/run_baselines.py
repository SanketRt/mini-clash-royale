#!/usr/bin/env python3
import os, sys
import numpy as np
import csv

proj_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from envs.mini_clash import MiniClashEnv
from scripts.logger import CSVLogger


def run_and_log(agent_name, policy_fn, env, episodes=100):
    logger = CSVLogger(out_dir="results",
                       filename=f"{agent_name}.csv",
                       fieldnames=["episode", "reward", "win", "steps"])
    for ep in range(1, episodes+1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

        win = int(env.opp_hp <= 0)
        logger.log(episode=ep,
                   reward=total_reward,
                   win=win,
                   steps=steps)
    with open(logger.path, newline='') as f:
        reader = csv.DictReader(f)
        rewards = [float(row["reward"]) for row in reader]
    avg = np.mean(rewards)
    print(f"[{agent_name}] Avg reward over {episodes} episodes: {avg:.2f}")

def cost_priority(obs):
    elixir = obs[2]
    if   elixir >= 6: return 3
    elif elixir >= 4: return 2
    elif elixir >= 2: return 1
    else: return 0

def cheap_swarm(obs):
    # obs = [your_hp, opp_hp, your_elixir, opp_elixir, ...]
    your_elixir = obs[2]
    return 1 if your_elixir >= 2 else 0

def mixed_policy(obs):
    elixir = obs[2]
    step   = env.step_count  # you’ll need to capture `env` in outer scope
    # Every 5th step: go all-in on the big unit
    if step % 5 == 0 and elixir >= 6:
        return 3
    # Otherwise, keep up a steady stream of cheap units
    return 1 if elixir >= 2 else 0


if __name__ == "__main__":
    env = MiniClashEnv()

    print("\n--- Random policy ---")
    run_and_log(
        "random",
        policy_fn=lambda obs: env.action_space.sample(),
        env=env,
        episodes=200
    )

    print("\n--- Cost priority ---")
    run_and_log("cost_priority", cost_priority, env, episodes=200)

    print("\n--- Cheap swarm ---")
    run_and_log("cheap_swarm", cheap_swarm, env, episodes=200)
    
    print("\n--- Mixed policy ---")
    run_and_log("mixed_policy", mixed_policy, env, episodes=200)

    print("\n--- Simple scripted policy ---")
    def scripted(obs):
        # spawn type B when you have ≥4 elixir, else pass
        return 2 if obs[2] >= 4 else 0

    run_and_log(
        "scripted",
        policy_fn=scripted,
        env=env,
        episodes=200
    )

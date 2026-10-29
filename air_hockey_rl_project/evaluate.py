import argparse
import numpy as np
from air_hockey_env import AirHockeyEnv
from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/air_hockey_ppo.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    model = PPO.load(args.model)
    env = AirHockeyEnv(opponent_level="medium", max_steps=2400)
    wins = losses = draws = 0
    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed+ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
        returns.append(total)
        if info.get("goal") == "agent": wins += 1
        elif info.get("goal") == "opponent": losses += 1
        else: draws += 1
    print(f"Episodes: {args.episodes}")
    print(f"Wins/losses/timeouts: {wins}/{losses}/{draws}")
    print(f"Win rate: {wins/args.episodes:.3f}")
    print(f"Mean return: {np.mean(returns):.3f} +/- {np.std(returns):.3f}")
    env.close()

if __name__ == "__main__":
    main()

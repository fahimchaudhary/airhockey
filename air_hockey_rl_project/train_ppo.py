import argparse, os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from air_hockey_env import AirHockeyEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="models/air_hockey_ppo")
    args = parser.parse_args()
    env = AirHockeyEnv(opponent_level="medium", max_steps=2400, randomize=True)
    check_env(env, warn=True)
    env = Monitor(env)
    model = PPO(
        "MlpPolicy", env, seed=args.seed, verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, policy_kwargs=dict(net_arch=dict(pi=[128,128], vf=[128,128])),
        tensorboard_log="runs/air_hockey/"
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}.zip")
    env.close()

if __name__ == "__main__":
    main()

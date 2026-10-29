# Air Hockey RL Lab

A compact, research-oriented 2D air-hockey project with:
- Vertical, top-down Pygame game
- Human vs AI play, plus Easy / Medium / Hard rule-based difficulty
- Gymnasium-compatible continuous-control environment
- PPO training with Stable-Baselines3
- Evaluation script for reproducible episodes

## 1. Requirements

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Play the game

```bash
python play_game.py
```

Controls:
- Mouse/touchpad: move your paddle by moving the pointer in the lower half.
- W/A/S/D or arrow keys: alternative movement.
- R: restart match
- 1 / 2 / 3: Easy / Medium / Hard rule-based AI
- ESC: quit

The board is portrait-oriented, with the human player at the bottom and the opponent at the top.

## 3. Train a PPO agent

```bash
python train_ppo.py --timesteps 300000
```

The model is saved to `models/air_hockey_ppo.zip`.

Training runs headlessly and does not open a game window. Start with 100,000–300,000 steps to test the pipeline; meaningful performance may require more steps and tuning.

## 4. Play against the trained PPO agent

```bash
python play_game.py --ai ppo --model models/air_hockey_ppo.zip
```

If the model file does not exist, use `--ai rule --difficulty medium`.

## 5. Evaluate a trained model

```bash
python evaluate.py --model models/air_hockey_ppo.zip --episodes 100
```

## Research framing

This is a lightweight demonstration environment, not a validated reproduction of the NeurIPS Robot Air Hockey Challenge. It uses simplified planar dynamics and does not model robot-arm kinematics, torque limits, real sensors, sim-to-real transfer, or hardware safety.

Suggested experiments:
1. PPO vs rule-based opponents across 5 random seeds.
2. Compare easy-to-hard curriculum against training only on hard opponents.
3. Add randomized puck friction, restitution, and observation noise.
4. Compare PPO with SAC using the same environment, reward, time budget, and seeds.
5. Report mean return, win rate, goal difference, episode length, and confidence intervals.

## Environment contract

- Observation: normalized vector of puck position/velocity, agent paddle position, opponent paddle position, and score difference.
- Action: two continuous values in [-1, 1], interpreted as desired paddle velocity.
- Reward: +1 for scoring, -1 for conceding, small shaping reward for approaching/controlling the puck, and a small movement penalty.
- Episodes terminate after a goal; time limit truncation is handled by the trainer/wrapper.

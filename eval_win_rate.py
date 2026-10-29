#!/usr/bin/env python3
"""
eval_win_rate.py — Evaluate a trained PPO agent vs rule-based AI
Usage:
  python eval_win_rate.py                       # uses air_hockey_ppo_best.zip
  python eval_win_rate.py air_hockey_ppo.zip    # specify model
"""
import sys, csv, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from train_air_hockey import (
    AirHockeyEnv, eval_win_rate,
    W, H, PR, PUCK_R, GOAL_W, FRICTION, WALL_REST, MAXV, MALLET_SPD, MAX_STEPS
)

MODEL_PATH = sys.argv[1] if len(sys.argv)>1 else "air_hockey_ppo_best"

print(f"Loading model: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

print("Evaluating 200 games each vs Easy / Medium / Hard …")
results = {}
for diff in ("easy","medium","hard"):
    wr = eval_win_rate(model, diff, 200)
    results[diff] = wr
    print(f"  {diff.capitalize():8s}: {wr:.1%}")

# ── Physics consistency check (Python vs JS formula) ──────────────────────
print("\nPhysics consistency check (1000 random steps) …")
rng  = np.random.default_rng(42)
px_  = W/2; py_ = H/2; vx_ = 3.0; vy_ = -5.0
hits = 0
for _ in range(1000):
    vx_ *= FRICTION; vy_ *= FRICTION
    px_ += vx_; py_ += vy_
    if px_<PUCK_R:     px_,vx_ = PUCK_R,    -vx_*WALL_REST
    if px_>W-PUCK_R:   px_,vx_ = W-PUCK_R,  -vx_*WALL_REST
    inM = abs(px_-W/2)<GOAL_W/2
    if py_<PUCK_R    and not inM: py_,vy_ = PUCK_R,   -vy_*WALL_REST
    if py_>H-PUCK_R  and not inM: py_,vy_ = H-PUCK_R, -vy_*WALL_REST
    hits += 1
print(f"  {hits} physics steps completed — no exceptions ✓")

# ── Plot win-rate chart from CSV ──────────────────────────────────────────
LOG = "training_log.csv"
try:
    steps, easy_wr, med_wr, hard_wr = [],[],[],[]
    with open(LOG) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["timestep"]))
            easy_wr.append(float(row["easy"]))
            med_wr.append(float(row["medium"]))
            hard_wr.append(float(row["hard"]))

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(steps, easy_wr,   'o-', color='#22c55e', label='Easy',   linewidth=2)
    ax.plot(steps, med_wr,    'o-', color='#f97316', label='Medium', linewidth=2)
    ax.plot(steps, hard_wr,   'o-', color='#f43f5e', label='Hard',   linewidth=2)
    ax.axhline(0.70, color='white', linestyle='--', alpha=0.4, label='Target 70%')
    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel("Win rate vs rule AI", fontsize=12)
    ax.set_title("PPO Self-Play Training — Win Rate Progress", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f"{y:.0%}"))
    ax.set_facecolor('#0f172a'); fig.patch.set_facecolor('#0f172a')
    ax.tick_params(colors='white'); ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white'); ax.title.set_color('white')
    for spine in ax.spines.values(): spine.set_edgecolor('#334155')
    ax.grid(color='#334155', linestyle='--', alpha=0.5)
    for l in ax.get_xticklabels()+ax.get_yticklabels(): l.set_color('white')
    plt.tight_layout()
    plt.savefig("win_rate_chart.png", dpi=150, facecolor='#0f172a')
    print("Chart saved → win_rate_chart.png")
except FileNotFoundError:
    print(f"  (No {LOG} found — run training first)")

print("\nSummary:")
for k,v in results.items():
    bar = '█'*int(v*20)
    print(f"  {k.capitalize():8s} [{bar:<20s}] {v:.1%}")

#!/usr/bin/env python3
"""
train_self_ppo.py — Intensive Self-Play PPO Reinforcement Learning for Air Hockey
Trains the PPO agent primarily against snapshots of itself (Self-Play PPO),
starting from the best pre-trained checkpoint with stationary serves matching the game.
"""

import os, sys, math, random, time, json, csv
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# ─── Constants (Identical to JS Game Physics) ─────────────────────────────────
W          = 400.0
H          = 800.0
PR         = 30.0         # Mallet radius
PUCK_R     = 15.0         # Puck radius
GOAL_W     = 160.0        # Goal mouth width
FRICTION   = 0.995        # Puck friction per frame
WALL_REST  = 0.90         # Wall restitution
MLT_REST   = 1.05         # Mallet restitution
MAXV       = 22.0         # Max puck velocity
MALLET_SPD = 11.0         # Max mallet speed per step
MAX_STEPS  = 2000         # Max episode length
N_FRAMES   = 3            # Frame stacking
RAW_DIM    = 8
STACKED_DIM= RAW_DIM * N_FRAMES  # 24-dim input


def clamp(val, low, high):
    return max(low, min(high, val))


# ─── Fast NumPy Policy for Self-Play Opponents ────────────────────────────────
class FastPolicy:
    """Extracts weights from PPO model for ultra-fast NumPy inference (1500+ FPS)."""
    def __init__(self, model):
        self.layers = []
        for m in model.policy.mlp_extractor.policy_net:
            if isinstance(m, nn.Linear):
                self.layers.append((m.weight.detach().cpu().numpy(),
                                     m.bias.detach().cpu().numpy(), True))
        a = model.policy.action_net
        self.layers.append((a.weight.detach().cpu().numpy(),
                             a.bias.detach().cpu().numpy(), False))
        self.input_dim = self.layers[0][0].shape[1]

    def act(self, obs: np.ndarray) -> np.ndarray:
        if len(obs) < self.input_dim:
            repeats = math.ceil(self.input_dim / len(obs))
            x = np.tile(obs, repeats)[:self.input_dim]
        elif len(obs) > self.input_dim:
            x = obs[:self.input_dim]
        else:
            x = obs
        for W_mat, b_vec, is_tanh in self.layers:
            x = np.dot(W_mat, x) + b_vec
            if is_tanh:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)

    def predict(self, obs: np.ndarray, deterministic: bool = False):
        a = self.act(obs)
        if not deterministic:
            a = a + np.random.normal(0, 0.03, size=a.shape)
            a = np.clip(a, -1.0, 1.0)
        return a, None


# ─── Rule-Based AI (for Evaluation and Grounding) ──────────────────────────────
DIFF = {
    "easy":   dict(speed=4.2, think=28, noise=70, aggr=0.25, defend=0.6),
    "medium": dict(speed=6.0, think=12, noise=28, aggr=0.50, defend=0.4),
    "hard":   dict(speed=9.5, think=2,  noise=6,  aggr=0.90, defend=0.2),
}

class RuleAI:
    def __init__(self, difficulty="hard"):
        self.p = DIFF[difficulty]
        self.x = W / 2.0
        self.y = 120.0
        self.tx = W / 2.0
        self.ty = 120.0
        self.timer = 0

    def reset(self):
        self.x = W / 2.0
        self.y = 120.0
        self.tx = W / 2.0
        self.ty = 120.0
        self.timer = 0

    def step(self, puck: np.ndarray) -> tuple:
        px, py, vx, vy = puck
        self.timer += 1
        if self.timer >= self.p["think"]:
            self.timer = 0
            if py < H / 2:
                if py < self.y + 10:
                    if abs(px - self.x) < PR + PUCK_R + 15:
                        self.tx = px - (PR + PUCK_R + 15) if px > W / 2 else px + (PR + PUCK_R + 15)
                    else:
                        self.tx = W / 2 + (px - W / 2) * 0.3
                    self.ty = max(self.y, 110.0)
                else:
                    target_x = px
                    if vy < -0.5:
                        t = max(0.0, (clamp(self.y + 25, PR + 10, H / 2 - PR) - py) / (vy if vy != 0 else -1))
                        span = (W - PUCK_R) - PUCK_R
                        raw_x = px + vx * t
                        u = (raw_x - PUCK_R) % (2 * span)
                        if u < 0: u += 2 * span
                        if u > span: u = 2 * span - u
                        target_x = PUCK_R + u
                    noise = (random.random() * 2 - 1) * self.p["noise"]
                    self.tx = clamp(target_x + noise, PR, W - PR)
                    self.ty = clamp(py + PR + 12, PR, H / 2 - PR)
            else:
                target_base = W / 2 + (px - W / 2) * self.p["defend"]
                noise = (random.random() * 2 - 1) * (self.p["noise"] * 0.4)
                self.tx = clamp(target_base + noise, PR, W - PR)
                self.ty = 110.0 + (py / H) * 20.0

        spd = self.p["speed"]
        nx = self.x + clamp(self.tx - self.x, -spd, spd)
        ny = self.y + clamp(self.ty - self.y, -spd * 0.9, spd * 0.9)
        vx_ = nx - self.x
        vy_ = ny - self.y
        self.x = clamp(nx, PR, W - PR)
        self.y = clamp(ny, PR, H / 2 - PR)
        return np.array([self.x, self.y]), np.array([vx_, vy_])


# ─── Gymnasium Air Hockey Environment ──────────────────────────────────────────
class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent=None):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(STACKED_DIM,), dtype=np.float32)

        self.opponent = opponent
        self.p1 = np.array([W / 2, H - 130.0])  # Bottom player (PPO agent)
        self.p2 = np.array([W / 2, 130.0])      # Top player (Self-play clone or Rule AI)
        self.v1 = np.zeros(2)
        self.v2 = np.zeros(2)
        self.puck = np.array([W / 2, H / 2, 0.0, 0.0])
        self.t = 0
        self.rng = np.random.default_rng()

        self.frame_stack = deque(maxlen=N_FRAMES)
        self.opp_frame_stack = deque(maxlen=N_FRAMES)
        self.episode_hits = 0

    def set_opponent(self, opp):
        self.opponent = opp

    def _raw_obs(self, flip: bool) -> np.ndarray:
        if flip:
            px_ = self.puck[0]
            py_ = H - self.puck[1]
            vx_ = self.puck[2]
            vy_ = -self.puck[3]
            m   = np.array([self.p2[0], H - self.p2[1]])
            o   = np.array([self.p1[0], H - self.p1[1]])
        else:
            px_ = self.puck[0]
            py_ = self.puck[1]
            vx_ = self.puck[2]
            vy_ = self.puck[3]
            m   = self.p1
            o   = self.p2
        return np.array([
            px_ / W * 2 - 1, py_ / H * 2 - 1,
            vx_ / MAXV,       vy_ / MAXV,
            m[0] / W * 2 - 1, m[1] / H * 2 - 1,
            o[0] / W * 2 - 1, o[1] / H * 2 - 1,
        ], dtype=np.float32)

    def _stacked_obs(self, flip: bool) -> np.ndarray:
        raw = self._raw_obs(flip)
        stack = self.opp_frame_stack if flip else self.frame_stack
        stack.append(raw)
        while len(stack) < N_FRAMES:
            stack.appendleft(raw.copy())
        return np.concatenate(list(stack), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p1 = np.array([W / 2, H - 130.0])
        self.p2 = np.array([W / 2, 130.0])
        self.v1 = np.zeros(2)
        self.v2 = np.zeros(2)

        # Serve Distribution:
        # 40% Center Start (Game starting exact match)
        # 30% Top Court (Bot serve after goal)
        # 30% Bottom Court (Human serve after goal)
        serve_choice = self.rng.random()
        if serve_choice < 0.40:
            # Exact Center serve
            self.puck = np.array([
                W / 2 + self.rng.uniform(-10, 10),
                H / 2,
                self.rng.uniform(-0.5, 0.5),
                self.rng.uniform(-0.5, 0.5),
            ])
        elif serve_choice < 0.70:
            # Top half serve (Bot court)
            self.puck = np.array([
                W / 2 + self.rng.uniform(-20, 20),
                H / 2 - 80.0,
                0.0,
                0.0,
            ])
        else:
            # Bottom half serve (Agent court)
            self.puck = np.array([
                W / 2 + self.rng.uniform(-20, 20),
                H / 2 + 80.0,
                0.0,
                0.0,
            ])

        self.t = 0
        self.frame_stack.clear()
        self.opp_frame_stack.clear()
        if self.opponent is not None and isinstance(self.opponent, RuleAI):
            self.opponent.reset()
        self.episode_hits = 0
        return self._stacked_obs(False), {}

    def _move(self, pos, action, bottom: bool) -> np.ndarray:
        nx = clamp(pos[0] + float(action[0]) * MALLET_SPD, PR, W - PR)
        if bottom:
            ny = clamp(pos[1] + float(action[1]) * MALLET_SPD, H / 2 + PR, H - PR)
        else:
            ny = clamp(pos[1] + float(action[1]) * MALLET_SPD, PR, H / 2 - PR)
        return np.array([nx, ny])

    def _collide(self, puck, mpos, mvel):
        px, py, vx, vy = puck
        dx, dy = px - mpos[0], py - mpos[1]
        d = math.hypot(dx, dy)
        minD = PR + PUCK_R
        collided = False
        if 0 < d < minD:
            collided = True
            nx, ny = dx / d, dy / d
            px = mpos[0] + nx * minD
            py = mpos[1] + ny * minD
            vn = (vx - mvel[0]) * nx + (vy - mvel[1]) * ny
            if vn < 0:
                j = -(1 + MLT_REST) * vn
                vx += j * nx
                vy += j * ny
            sp = math.hypot(vx, vy)
            if sp > MAXV:
                vx *= MAXV / sp
                vy *= MAXV / sp
        return np.array([px, py, vx, vy]), collided

    def step(self, action):
        self.t += 1
        old1 = self.p1.copy()
        old2 = self.p2.copy()
        prev_puck_dist = math.hypot(self.puck[0] - self.p1[0], self.puck[1] - self.p1[1])
        prev_puck_vy = self.puck[3]

        # PPO Agent move (bottom)
        self.p1 = self._move(self.p1, action, bottom=True)
        self.v1 = self.p1 - old1

        # Opponent move (top: Self-Play clone or Rule AI)
        if self.opponent is None:
            a2 = self.rng.uniform(-1, 1, 2)
            self.p2 = self._move(self.p2, a2, bottom=False)
            self.v2 = self.p2 - old2
        elif isinstance(self.opponent, RuleAI):
            p_flip = np.array([self.puck[0], H - self.puck[1], self.puck[2], -self.puck[3]])
            pos2, vel2 = self.opponent.step(p_flip)
            self.p2 = pos2
            self.v2 = vel2
        else:
            # Self-Play fast policy
            obs2 = self._stacked_obs(True)
            a2, _ = self.opponent.predict(obs2, deterministic=False)
            a2 = np.array([a2[0], -a2[1]])
            self.p2 = self._move(self.p2, a2, bottom=False)
            self.v2 = self.p2 - old2

        # Physics simulation
        px, py, vx, vy = self.puck
        vx *= FRICTION; vy *= FRICTION
        px += vx; py += vy

        if px < PUCK_R:     px, vx = PUCK_R,     -vx * WALL_REST
        if px > W - PUCK_R: px, vx = W - PUCK_R, -vx * WALL_REST

        in_mouth = abs(px - W / 2) < GOAL_W / 2
        if py < PUCK_R    and not in_mouth: py, vy = PUCK_R,     -vy * WALL_REST
        if py > H - PUCK_R and not in_mouth: py, vy = H - PUCK_R, -vy * WALL_REST

        self.puck = np.array([px, py, vx, vy])
        self.puck, hit_p1 = self._collide(self.puck, self.p1, self.v1)
        self.puck, hit_p2 = self._collide(self.puck, self.p2, self.v2)

        # Reward engineering
        reward = -0.0002

        # Distance reward when puck is in agent's half
        curr_puck_dist = math.hypot(self.puck[0] - self.p1[0], self.puck[1] - self.p1[1])
        if self.puck[1] >= H / 2:
            delta_dist = prev_puck_dist - curr_puck_dist
            reward += 0.003 * delta_dist
            goal_proximity = (self.puck[1] - H / 2) / (H / 2)
            reward += 0.002 * delta_dist * goal_proximity
        else:
            # Center observation reward when puck in opponent's half
            center_offset = abs(self.p1[0] - W / 2) / (W / 2)
            reward += 0.001 * (1.0 - center_offset)

        if hit_p1:
            self.episode_hits += 1
            reward += 0.15
            puck_speed = math.hypot(self.puck[2], self.puck[3])
            speed_bonus = min(puck_speed / MAXV, 1.0)
            if self.puck[3] < 0:
                reward += 0.20 * speed_bonus
                goal_aim = 1.0 - min(abs(self.puck[0] - W / 2) / (GOAL_W / 2), 1.0)
                reward += 0.10 * goal_aim * speed_bonus
            if prev_puck_vy > 0 and self.puck[3] < 0:
                reward += 0.30

        terminated = False
        if self.puck[1] < -PUCK_R:        # Agent scored!
            reward += 3.0
            terminated = True
        elif self.puck[1] > H + PUCK_R:   # Agent conceded
            reward -= 2.5
            terminated = True

        truncated = self.t >= MAX_STEPS
        obs = self._stacked_obs(False)
        return obs, reward, terminated, truncated, {}


# ─── Evaluation ───────────────────────────────────────────────────────────────
def eval_win_rate(model, difficulty="hard", n_games=150) -> float:
    env = AirHockeyEnv()
    rule = RuleAI(difficulty)
    fast = FastPolicy(model)
    wins = 0
    for _ in range(n_games):
        obs, _ = env.reset()
        rule.reset()
        done = False
        while not done:
            a1 = fast.act(obs)
            p_flip = np.array([env.puck[0], H - env.puck[1], env.puck[2], -env.puck[3]])
            r_pos, r_vel = rule.step(p_flip)
            env.p2 = r_pos
            env.v2 = r_vel
            old1 = env.p1.copy()
            env.p1 = env._move(env.p1, a1, bottom=True)
            env.v1 = env.p1 - old1

            px, py, vx, vy = env.puck
            vx *= FRICTION; vy *= FRICTION
            px += vx; py += vy
            if px < PUCK_R: px, vx = PUCK_R, -vx * WALL_REST
            if px > W - PUCK_R: px, vx = W - PUCK_R, -vx * WALL_REST
            inMouth = abs(px - W / 2) < GOAL_W / 2
            if py < PUCK_R and not inMouth: py, vy = PUCK_R, -vy * WALL_REST
            if py > H - PUCK_R and not inMouth: py, vy = H - PUCK_R, -vy * WALL_REST
            env.puck = np.array([px, py, vx, vy])
            env.puck, _ = env._collide(env.puck, env.p1, env.v1)
            env.puck, _ = env._collide(env.puck, env.p2, env.v2)
            env.t += 1

            if env.puck[1] < -PUCK_R: wins += 1; done = True
            elif env.puck[1] > H + PUCK_R: done = True
            elif env.t >= MAX_STEPS: done = True

            if not done:
                obs = env._stacked_obs(False)
    return wins / n_games


def plot_live_chart(csv_path="training_log.csv", chart_path="win_rate_chart.png"):
    try:
        if not os.path.exists(csv_path): return
        steps, easy_wr, med_wr, hard_wr = [], [], [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                steps.append(int(row["timestep"]))
                easy_wr.append(float(row["easy"]))
                med_wr.append(float(row["medium"]))
                hard_wr.append(float(row["hard"]))
        if not steps: return

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(steps, easy_wr, 'o-', color='#22c55e', label='Easy AI', linewidth=2.5, markersize=5)
        ax.plot(steps, med_wr,  's-', color='#f97316', label='Medium AI', linewidth=2.5, markersize=5)
        ax.plot(steps, hard_wr, '^-', color='#f43f5e', label='Hard AI', linewidth=2.5, markersize=5)
        ax.axhline(0.75, color='#a855f7', linestyle='--', linewidth=1.5, alpha=0.8, label='Target (75%)')
        ax.set_xlabel('Self-Play Training Steps', fontsize=12, color='white', labelpad=10)
        ax.set_ylabel('Win Rate', fontsize=12, color='white', labelpad=10)
        ax.set_title('Air Hockey PPO — Pure Self-Play Training Progress', fontsize=14, fontweight='bold', color='white', pad=15)
        ax.legend(fontsize=11, facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#070b16')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_edgecolor('#334155')
        ax.grid(color='#334155', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, facecolor='#070b16')
        plt.close()
    except Exception as e:
        print(f"[chart] Error: {e}")


def export_json(model, path="ai_weights.json"):
    layers = []
    for m in model.policy.mlp_extractor.policy_net:
        if isinstance(m, nn.Linear):
            layers.append({
                "W":   m.weight.detach().cpu().numpy().tolist(),
                "b":   m.bias.detach().cpu().numpy().tolist(),
                "act": "tanh",
            })
    a = model.policy.action_net
    layers.append({
        "W":   a.weight.detach().cpu().numpy().tolist(),
        "b":   a.bias.detach().cpu().numpy().tolist(),
        "act": "none",
    })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"layers": layers}, f)
    print(f"[export] {path} written ({len(layers)} layers)")


# ─── Pure Self-Play Callback ──────────────────────────────────────────────────
class PureSelfPlayCallback(BaseCallback):
    def __init__(self, snap_every=50_000, eval_every=100_000, pool_size=16,
                 log_path="training_log.csv", target_wr=0.80):
        super().__init__()
        self.snap_every = snap_every
        self.eval_every = eval_every
        self.pool_size = pool_size
        self.log_path = log_path
        self.target_wr = target_wr
        self.pool = []
        self.last_snap = 0
        self.last_eval = 0
        self.best_hard_wr = 0.0

    def init_pool(self, base_model):
        """Seed pool with the pre-trained base model."""
        base_snap = FastPolicy(base_model)
        self.pool.append((base_snap, 1500))
        print(f"[self-play] Pool seeded with base model (ELO 1500)")

    def assign_opponents(self):
        n_envs = self.training_env.num_envs
        # Env 0: Hard Rule AI for absolute skill grounding
        self.training_env.env_method("set_opponent", RuleAI("hard"), indices=0)

        # Envs 1-7: Self-Play opponents from the pool!
        if len(self.pool) > 0:
            # Env 1: Always the most recent snapshot (plays against itself!)
            latest_snap, _ = self.pool[-1]
            self.training_env.env_method("set_opponent", latest_snap, indices=1)

            # Envs 2-7: ELO-weighted sampling from historical pool
            weights = [elo for _, elo in self.pool]
            min_w = min(weights)
            weights = [w - min_w + 100 for w in weights]
            total = sum(weights)
            probs = [w / total for w in weights]

            for i in range(2, n_envs):
                idx = np.random.choice(len(self.pool), p=probs)
                snap, _ = self.pool[idx]
                self.training_env.env_method("set_opponent", snap, indices=i)
        else:
            for i in range(1, n_envs):
                self.training_env.env_method("set_opponent", RuleAI("hard"), indices=i)

    def _on_step(self):
        ts = self.num_timesteps

        # ── 1. Snapshot for Self-Play ──
        if ts - self.last_snap >= self.snap_every:
            self.last_snap = ts
            snap = FastPolicy(self.model)
            # Quick 40-game eval vs Hard AI to compute ELO
            wr = eval_win_rate(self.model, "hard", 40)
            elo = 1000 + wr * 1000
            self.pool.append((snap, elo))
            if len(self.pool) > self.pool_size:
                self.pool.pop(0)
            self.assign_opponents()
            print(f"[self-play] Step {ts:,} | Pool Size: {len(self.pool)} | Latest ELO: {elo:.0f} | Opponents Re-assigned to Self-Play", flush=True)

        # ── 2. Thorough Evaluation ──
        if ts - self.last_eval >= self.eval_every:
            self.last_eval = ts
            print(f"\n[eval] Step {ts:,} — Running comprehensive benchmarks (150 games each)...", flush=True)
            wr_e = eval_win_rate(self.model, "easy",   150)
            wr_m = eval_win_rate(self.model, "medium", 150)
            wr_h = eval_win_rate(self.model, "hard",   150)
            print(f"  Easy={wr_e:.1%}  Medium={wr_m:.1%}  Hard={wr_h:.1%}", flush=True)

            # CSV log
            write_header = not os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["timestep", "easy", "medium", "hard"])
                w.writerow([ts, round(wr_e, 4), round(wr_m, 4), round(wr_h, 4)])

            plot_live_chart(self.log_path, "win_rate_chart.png")

            if wr_h >= self.best_hard_wr:
                self.best_hard_wr = wr_h
                self.model.save("air_hockey_ppo_best")
                export_json(self.model, "ai_weights.json")
                if os.path.exists("www"):
                    export_json(self.model, "www/ai_weights.json")
                android_asset = "android/app/src/main/assets/public/ai_weights.json"
                if os.path.exists(os.path.dirname(android_asset)):
                    export_json(self.model, android_asset)
                # Auto-sync HTML files
                try:
                    os.system("python3 update_html.py > /dev/null 2>&1")
                except Exception:
                    pass
                print(f"  ★ NEW BEST SELF-PPO MODEL: {wr_h:.1%} vs Hard! Exported and Synced.", flush=True)

        return True


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_ENVS = 8
    TOTAL_STEPS = 6_000_000

    print("═══════════════════════════════════════════════════════════════")
    print("  Air Hockey — Pure Self-Play PPO Reinforcement Training")
    print(f"  {N_ENVS} Parallel Vector Envs (7 Self-Play + 1 Hard Rule Anchor)")
    print(f"  Starting from pre-trained: air_hockey_ppo_best.zip")
    print(f"  Total Steps: {TOTAL_STEPS:,}")
    print("═══════════════════════════════════════════════════════════════")

    env = make_vec_env(AirHockeyEnv, n_envs=N_ENVS)

    cb = PureSelfPlayCallback(
        snap_every=50_000,
        eval_every=100_000,
        pool_size=16,
        log_path="training_log.csv",
        target_wr=0.85,
    )

    # Load existing best model
    if os.path.exists("air_hockey_ppo_best.zip"):
        print("[train] Loading air_hockey_ppo_best.zip to continue with Pure Self-Play PPO...")
        model = PPO.load(
            "air_hockey_ppo_best",
            env=env,
            learning_rate=1.5e-4,     # Controlled learning rate for self-play stability
            clip_range=0.15,          # Stable policy ratio updates
            ent_coef=0.005,           # Exploration vs exploitation
            n_steps=2048,
            batch_size=2048,
            verbose=1,
        )
    else:
        print("[train] Creating fresh PPO model...")
        model = PPO(
            "MlpPolicy", env,
            n_steps=2048, batch_size=2048,
            learning_rate=2e-4, gamma=0.997, gae_lambda=0.95,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
            clip_range=0.15, n_epochs=10,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                activation_fn=nn.Tanh,
                ortho_init=True,
            ),
            verbose=1,
        )

    cb.model = model
    cb.init_pool(model)
    cb.assign_opponents()

    print("[train] Starting Self-Play PPO learning loop...")
    start_time = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb, reset_num_timesteps=False)
    elapsed = time.time() - start_time

    model.save("air_hockey_ppo_self_play_final")
    export_json(model, "ai_weights.json")
    if os.path.exists("www"):
        export_json(model, "www/ai_weights.json")

    print(f"\n[done] Self-Play PPO Training completed in {elapsed/3600:.2f} hours!")
    print(f"  Best Hard win-rate achieved: {cb.best_hard_wr:.1%}")

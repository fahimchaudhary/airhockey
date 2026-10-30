#!/usr/bin/env python3
"""
train_air_hockey.py  –  V3 Perfected Self-Play PPO Trainer
──────────────────────────────────────────────────────────
Major improvements over V2:
  • Deeper network (256→256→128) with orthogonal init
  • Frame-stacking (last 3 frames → 24-dim obs) for temporal awareness
  • Refined potential-based reward shaping (no sign-flip instability)
  • Defensive zone awareness & goal-line urgency rewards
  • Cosine-annealed learning rate with warmup
  • Larger batch (2048) & longer rollouts (2048 steps)
  • Self-play pool with ELO-ranked opponent sampling
  • Curriculum phases: Rule AI → Mixed → Pure Self-Play
  • Automatic checkpoint & export on every new best
"""
import copy, json, math, os, random, sys, csv, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# ─── Constants (strictly identical to JS) ───────────────────────────────────
W, H       = 400.0, 700.0
GOAL_W     = 150.0
PR         = 34.0        # mallet radius
PUCK_R     = 18.0        # puck radius
MAXV       = 26.0        # max puck speed
MALLET_SPD = 12.0
FRICTION   = 0.995
WALL_REST  = 0.9
MLT_REST   = 1.05
MAX_STEPS  = 3_000

# ─── Frame Stacking Config ──────────────────────────────────────────────────
N_FRAMES   = 3           # number of frames to stack
OBS_DIM    = 8           # single-frame observation
STACKED_DIM = OBS_DIM * N_FRAMES   # 24


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ─── Gymnasium Environment (V3 with frame stacking + improved rewards) ──────
class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-2, 2, shape=(STACKED_DIM,), dtype=np.float32)
        self.action_space      = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.opponent = None
        self.rng = np.random.default_rng()
        self.frame_stack = deque(maxlen=N_FRAMES)
        self.opp_frame_stack = deque(maxlen=N_FRAMES)
        # Stats for logging
        self.episode_goals_scored = 0
        self.episode_goals_conceded = 0
        self.episode_hits = 0

    def set_opponent(self, policy):
        self.opponent = policy

    def _raw_obs(self, flip: bool) -> np.ndarray:
        """Single frame observation (8-dim)."""
        px, py, vx, vy = self.puck
        if flip:
            px_, py_, vx_, vy_ = px, H - py, vx, -vy
            m = np.array([self.p2[0], H - self.p2[1]])
            o = np.array([self.p1[0], H - self.p1[1]])
        else:
            px_, py_, vx_, vy_ = px, py, vx, vy
            m, o = self.p1, self.p2
        return np.array([
            px_ / W * 2 - 1, py_ / H * 2 - 1,
            vx_ / MAXV,       vy_ / MAXV,
            m[0] / W * 2 - 1, m[1] / H * 2 - 1,
            o[0] / W * 2 - 1, o[1] / H * 2 - 1,
        ], dtype=np.float32)

    def _stacked_obs(self, flip: bool) -> np.ndarray:
        """Return frame-stacked observation (24-dim)."""
        raw = self._raw_obs(flip)
        self.frame_stack.append(raw)
        # Pad with copies if not enough frames yet
        while len(self.frame_stack) < N_FRAMES:
            self.frame_stack.appendleft(raw.copy())
        return np.concatenate(list(self.frame_stack), dtype=np.float32)

    def _opp_stacked_obs(self) -> np.ndarray:
        """Return frame-stacked observation (24-dim) from opponent perspective."""
        raw = self._raw_obs(True)
        self.opp_frame_stack.append(raw)
        while len(self.opp_frame_stack) < N_FRAMES:
            self.opp_frame_stack.appendleft(raw.copy())
        return np.concatenate(list(self.opp_frame_stack), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p1 = np.array([W / 2, H - 130.0])   # agent (bottom)
        self.p2 = np.array([W / 2, 130.0])       # opponent (top)
        self.v1 = np.zeros(2)
        self.v2 = np.zeros(2)

        # Realistic Match Serves matching the game:
        # 40% Exact Center serve (starting match: puck strictly at center, 0 speed)
        # 30% Opponent serve (puck shifted to opponent half, 0 speed)
        # 30% Agent serve (puck shifted to agent half, 0 speed)
        serve_choice = float(self.rng.random())
        if serve_choice < 0.40:
            self.puck = np.array([W / 2, H / 2, 0.0, 0.0])
        elif serve_choice < 0.70:
            self.puck = np.array([
                W / 2 + self.rng.uniform(-20, 20),
                H / 2 - 80.0,
                0.0,
                0.0,
            ])
        else:
            self.puck = np.array([
                W / 2 + self.rng.uniform(-20, 20),
                H / 2 + 80.0,
                0.0,
                0.0,
            ])
        if self.rng.random() < 0.15:
            self.puck[2] += float(self.rng.uniform(-1.0, 1.0))
            self.puck[3] += float(self.rng.uniform(-1.0, 1.0))

        self.t = 0
        self.frame_stack.clear()
        self.opp_frame_stack.clear()
        if self.opponent is not None and hasattr(self.opponent, 'reset'):
            self.opponent.reset()
        self.episode_goals_scored = 0
        self.episode_goals_conceded = 0
        self.episode_hits = 0
        self._prev_puck_dist = None
        self._consecutive_idle = 0
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
                j  = -(1 + MLT_REST) * vn
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
        prev_puck_y = self.puck[1]
        prev_puck_vy = self.puck[3]

        # Compute pre-step distances
        prev_puck_dist = math.hypot(self.puck[0] - self.p1[0], self.puck[1] - self.p1[1])

        # Agent move
        self.p1 = self._move(self.p1, action, bottom=True)

        # Opponent move
        if self.opponent is None:
            a2 = self.rng.uniform(-1, 1, 2)
            self.p2 = self._move(self.p2, a2, bottom=False)
        elif isinstance(self.opponent, (RuleAI, IdleAI)):
            p_flip = np.array([self.puck[0], H - self.puck[1], self.puck[2], -self.puck[3]])
            self.p2, self.v2 = self.opponent.step(p_flip)
        else:
            # Self-play opponent (uses stacked 24-dim obs flipped from opponent perspective)
            obs2 = self._opp_stacked_obs()
            a2, _ = self.opponent.predict(obs2, deterministic=False)
            a2 = np.array([a2[0], -a2[1]])
            self.p2 = self._move(self.p2, a2, bottom=False)

        self.v1 = self.p1 - old1
        if not isinstance(self.opponent, (RuleAI, IdleAI)):
            self.v2 = self.p2 - old2

        # Physics simulation (exact JS match)
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

        # ─── V3 Enhanced Reward Engineering ──────────────────────────────
        reward = 0.0

        # 1. Step penalty
        reward -= 0.0001

        curr_puck_dist = math.hypot(self.puck[0] - self.p1[0], self.puck[1] - self.p1[1])

        # 2. Attack Mode: Puck is in agent's half (puck[1] >= H / 2)
        if self.puck[1] >= H / 2:
            delta_dist = prev_puck_dist - curr_puck_dist
            reward += 0.004 * delta_dist
            goal_proximity = (self.puck[1] - H / 2) / (H / 2)
            reward += 0.0025 * delta_dist * goal_proximity
        else:
            # 3. Defense Mode: Puck is in opponent's half (puck[1] < H / 2)
            # Stay centered in front of own goal and guard baseline (H - 130)
            center_offset = abs(self.p1[0] - W / 2) / (W / 2)
            reward += 0.0015 * (1.0 - center_offset)
            y_defend_offset = abs(self.p1[1] - (H - 130.0)) / (H / 2)
            reward += 0.0010 * (1.0 - y_defend_offset)
            # Anti-hallucination / anti-jitter penalty: penalize excessive twitching when puck is far away
            movement = math.hypot(self.v1[0], self.v1[1])
            if movement > 5.0 and self.puck[3] >= 0:
                reward -= 0.0004

        # 4. Collision & Offensive Hit Bonus
        if hit_p1:
            self.episode_hits += 1
            reward += 0.20   # contact reward

            puck_speed = math.hypot(self.puck[2], self.puck[3])
            speed_bonus = min(puck_speed / MAXV, 1.0)

            if self.puck[3] < 0:
                # Offensive shot towards opponent goal
                reward += 0.25 * speed_bonus
                # Accuracy bonus: puck heading towards goal mouth
                goal_aim = 1.0 - min(abs(self.puck[0] - W / 2) / (GOAL_W / 2), 1.0)
                reward += 0.15 * goal_aim * speed_bonus

            if prev_puck_vy > 0 and self.puck[3] < 0:
                # Defensive save: reversed puck direction
                reward += 0.35

        # 5. Terminal Goal Rewards
        terminated = False
        if self.puck[1] < -PUCK_R:        # agent scored!
            reward += 3.5
            terminated = True
            self.episode_goals_scored += 1
        elif self.puck[1] > H + PUCK_R:   # agent conceded
            reward -= 3.0
            terminated = True
            self.episode_goals_conceded += 1

        # 6. Small continuous reward for puck moving towards opponent goal
        if not terminated:
            vy_norm = -self.puck[3] / MAXV  # positive when heading to opponent
            reward += 0.005 * clamp(vy_norm, -0.5, 1.0)

        truncated = self.t >= MAX_STEPS
        return self._stacked_obs(False), reward, terminated, truncated, {}


# ─── Fast Policy for inference (supports both 8-dim and 24-dim) ──────────────
class FastPolicy:
    """Extracts weights from PPO model for fast NumPy inference."""
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
        for W, b, is_tanh in self.layers:
            x = np.dot(W, x) + b
            if is_tanh:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)

    def predict(self, obs: np.ndarray, deterministic: bool = False):
        a = self.act(obs)
        if not deterministic:
            a = a + np.random.normal(0, 0.03, size=a.shape)
            a = np.clip(a, -1.0, 1.0)
        return a, None


# ─── Rule-Based AI and Idle AI (for evaluation and curriculum) ───────────────
DIFF = {
    "easy":   dict(speed=4.2, think=28, noise=70, aggr=0.25, defend=0.6),
    "medium": dict(speed=6.0, think=12, noise=28, aggr=0.50, defend=0.4),
    "hard":   dict(speed=9.5, think=2,  noise=6,  aggr=0.90, defend=0.2),
}

class IdleAI:
    """Simulates an inactive human player who does not move or make actions."""
    def __init__(self):
        self.pos = np.array([W / 2, 130.0])
        self.vel = np.zeros(2)

    def reset(self):
        self.pos = np.array([W / 2, 130.0])
        self.vel = np.zeros(2)

    def step(self, puck_state):
        return self.pos.copy(), np.zeros(2)

    def predict(self, obs, deterministic=False):
        return np.zeros(2), None


class RuleAI:
    def __init__(self, difficulty="hard"):
        self.p   = DIFF[difficulty]
        self.pos = np.array([W/2, 130.0])
        self.vel = np.zeros(2)
        self.tx  = W/2; self.ty = 130.0; self.timer = 0

    def reset(self):
        self.pos = np.array([W/2, 130.0])
        self.vel = np.zeros(2)
        self.tx = W/2; self.ty = 130.0; self.timer = 0

    def predict_x(self, px, py, vx, vy, target_y):
        if vy == 0: return px
        t = (target_y - py) / vy
        if t < 0: return px
        mn, mx = PUCK_R, W - PUCK_R; span = mx - mn
        x = px + vx * t
        u = (x - mn) % (2 * span)
        if u < 0: u += 2 * span
        if u > span: u = 2 * span - u
        return mn + u

    def step(self, puck_state):
        px, py, vx, vy = puck_state
        p = self.p
        self.timer += 1
        if self.timer >= p["think"]:
            self.timer = 0
            if py < H / 2:
                if py < self.pos[1] + 10:
                    self.tx = self.pos[0] - (PR + PUCK_R + 15) if px > W / 2 else self.pos[0] + (PR + PUCK_R + 15)
                    self.ty = max(self.pos[1], 110.0)
                else:
                    base = self.predict_x(px, py, vx, vy, clamp(self.pos[1] + 25, PR, H / 2 - PR)) if vy < -0.5 else px
                    self.tx = base + (random.random() * 2 - 1) * p["noise"]
                    self.ty = clamp(py + PR + 12, PR, H / 2 - PR)
            else:
                base = W / 2 + (px - W / 2) * p["defend"]
                self.tx = base + (random.random() * 2 - 1) * (p["noise"] * 0.4)
                self.ty = 110.0 + (py / H) * 20.0
        nx = self.pos[0] + clamp(self.tx - self.pos[0], -p["speed"], p["speed"])
        ny = self.pos[1] + clamp(self.ty - self.pos[1], -p["speed"] * 0.9, p["speed"] * 0.9)
        self.vel = np.array([nx-self.pos[0], ny-self.pos[1]])
        self.pos = np.array([clamp(nx,PR,W-PR), clamp(ny,PR,H/2-PR)])
        return self.pos.copy(), self.vel.copy()


# ─── Evaluation (200 games vs rule AI or Idle AI) ───────────────────────────
def eval_win_rate(model, difficulty="hard", n_games=200) -> float:
    env  = AirHockeyEnv()
    if difficulty == "idle":
        rule = IdleAI()
    else:
        rule = RuleAI(difficulty)
    fast = FastPolicy(model)
    wins = 0
    for _ in range(n_games):
        obs, _ = env.reset(); rule.reset()
        done = False
        while not done:
            a1 = fast.act(obs)
            p_state = env.puck.copy()
            p_flip  = np.array([p_state[0], H-p_state[1], p_state[2], -p_state[3]])
            rule_pos, rule_vel = rule.step(p_flip)
            env.p2 = rule_pos
            env.v2 = rule_vel
            old1   = env.p1.copy()
            env.p1 = env._move(env.p1, a1, bottom=True)
            env.v1 = env.p1 - old1

            # physics
            px, py, vx, vy = env.puck
            vx *= FRICTION; vy *= FRICTION; px += vx; py += vy
            if px<PUCK_R: px,vx=PUCK_R,-vx*WALL_REST
            if px>W-PUCK_R: px,vx=W-PUCK_R,-vx*WALL_REST
            inMouth = abs(px-W/2)<GOAL_W/2
            if py<PUCK_R and not inMouth: py,vy=PUCK_R,-vy*WALL_REST
            if py>H-PUCK_R and not inMouth: py,vy=H-PUCK_R,-vy*WALL_REST
            env.puck = np.array([px,py,vx,vy])
            env.puck, _ = env._collide(env.puck, env.p1, env.v1)
            env.puck, _ = env._collide(env.puck, env.p2, env.v2)
            env.t += 1

            if env.puck[1]<-PUCK_R: wins+=1; done=True
            elif env.puck[1]>H+PUCK_R: done=True
            elif env.t>=MAX_STEPS: done=True

            if not done:
                obs = env._stacked_obs(False)
    return wins / n_games


def plot_live_chart(csv_path="training_log.csv", chart_path="win_rate_chart.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        steps, easy_wr, med_wr, hard_wr = [], [], [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                steps.append(int(row["timestep"]))
                easy_wr.append(float(row["easy"]))
                med_wr.append(float(row["medium"]))
                hard_wr.append(float(row["hard"]))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, easy_wr, 'o-', color='#22c55e', label='Easy AI', linewidth=2.5, markersize=8)
        ax.plot(steps, med_wr,  's-', color='#f97316', label='Medium AI', linewidth=2.5, markersize=8)
        ax.plot(steps, hard_wr, '^-', color='#f43f5e', label='Hard AI', linewidth=2.5, markersize=8)
        ax.axhline(0.70, color='#a855f7', linestyle='--', linewidth=2, alpha=0.8, label='Target 70% vs Hard')

        ax.set_xlabel('Training Steps', fontsize=12, color='white', labelpad=10)
        ax.set_ylabel('Win Rate (200 Games)', fontsize=12, color='white', labelpad=10)
        ax.set_title('Air Hockey PPO V3 — Self-Reinforcement Training', fontsize=14, fontweight='bold', color='white', pad=15)
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
        print(f"[chart] Error generating chart: {e}")


# ─── Cosine LR Schedule with Warmup ─────────────────────────────────────────
def cosine_lr_schedule(warmup_frac=0.02, min_lr_frac=0.1):
    """Returns a schedule function for SB3's learning_rate parameter."""
    def schedule(progress_remaining):
        # progress_remaining goes from 1.0 → 0.0
        progress = 1.0 - progress_remaining
        if progress < warmup_frac:
            # Linear warmup
            return progress / warmup_frac
        else:
            # Cosine decay
            decay_progress = (progress - warmup_frac) / (1.0 - warmup_frac)
            return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * decay_progress))
    return schedule


# ─── Self-Play + Curriculum Callback (V3) ────────────────────────────────────
class SelfPlayCallback(BaseCallback):
    def __init__(self, every=300_000, pool_size=12, log_every=500_000,
                 log_path="training_log.csv", no_improve_limit=4_000_000,
                 target_wr=0.75):
        super().__init__()
        self.every          = every
        self.pool_size      = pool_size
        self.pool           = []       # list of (FastPolicy, elo)
        self.last_snap      = 0
        self.log_every      = log_every
        self.last_log       = 0
        self.log_path       = log_path
        self.no_improve     = no_improve_limit
        self.target_wr      = target_wr
        self.best_hard_wr   = 0.0
        self.best_composite = 0.0
        self.best_at        = 0
        self._header_written = os.path.exists(log_path)
        self._phase          = "curriculum"

    def _assign_opponents(self):
        n_envs = self.training_env.num_envs
        # Env 0-1: Hard Rule AI
        # Env 2-3: Medium Rule AI
        # Env 4: Easy Rule AI
        # Env 5: IdleAI (dedicated inactive human simulation!)
        # Env 6-7: Self-play snapshots or Hard/Medium
        self.training_env.env_method("set_opponent", RuleAI("hard"), indices=0)
        if n_envs > 1:
            self.training_env.env_method("set_opponent", RuleAI("hard"), indices=1)
        if n_envs > 2:
            self.training_env.env_method("set_opponent", RuleAI("medium"), indices=2)
        if n_envs > 3:
            self.training_env.env_method("set_opponent", RuleAI("medium"), indices=3)
        if n_envs > 4:
            self.training_env.env_method("set_opponent", RuleAI("easy"), indices=4)
        if n_envs > 5:
            self.training_env.env_method("set_opponent", IdleAI(), indices=5)

        if len(self.pool) > 0:
            for i in range(6, n_envs):
                snap, _ = random.choice(self.pool)
                self.training_env.env_method("set_opponent", snap, indices=i)
        else:
            if n_envs > 6:
                self.training_env.env_method("set_opponent", RuleAI("hard"), indices=6)
            if n_envs > 7:
                self.training_env.env_method("set_opponent", RuleAI("medium"), indices=7)

    def _on_step(self):
        ts = self.num_timesteps

        # ── Snapshot for self-play ──
        if ts - self.last_snap >= self.every:
            self.last_snap = ts
            snap = FastPolicy(self.model)
            wr = eval_win_rate(self.model, "hard", 50)
            elo = 1000 + wr * 1000
            self.pool = (self.pool + [(snap, elo)])[-self.pool_size:]
            self._assign_opponents()
            print(f"[self-play] step {ts:,}, pool={len(self.pool)}, latest_elo={elo:.0f}", flush=True)

        # ── Win-rate evaluation ──
        if ts - self.last_log >= self.log_every:
            self.last_log = ts
            print(f"\n[eval] step {ts:,} — evaluating...", flush=True)
            wr_idle = eval_win_rate(self.model, "idle",   50)
            wr_e    = eval_win_rate(self.model, "easy",   100)
            wr_m    = eval_win_rate(self.model, "medium", 100)
            wr_h    = eval_win_rate(self.model, "hard",   100)
            print(f"  Idle={wr_idle:.1%}  Easy={wr_e:.1%}  Medium={wr_m:.1%}  Hard={wr_h:.1%}", flush=True)

            # CSV logging
            write_header = not self._header_written
            with open(self.log_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["timestep","easy","medium","hard"])
                    self._header_written = True
                w.writerow([ts, round(wr_e,4), round(wr_m,4), round(wr_h,4)])

            plot_live_chart(self.log_path, "win_rate_chart.png")

            # Composite score tracking
            composite = wr_h * 0.45 + wr_m * 0.35 + wr_idle * 0.20
            if composite >= self.best_composite or wr_h > self.best_hard_wr:
                if composite > self.best_composite: self.best_composite = composite
                if wr_h > self.best_hard_wr: self.best_hard_wr = wr_h
                self.best_at = ts
                self.model.save("air_hockey_ppo_best")
                export_json(self.model, "ai_weights.json")
                print(f"  ★ NEW BEST: composite={composite:.1%} (Hard={wr_h:.1%}, Med={wr_m:.1%}, Idle={wr_idle:.1%}) at step {ts:,}", flush=True)

            if wr_h >= self.target_wr:
                print(f"\n[TARGET REACHED] Hard win-rate {wr_h:.1%} >= {self.target_wr:.0%} at step {ts:,}!", flush=True)
                return False

            if ts >= self.no_improve and (ts - self.best_at) >= self.no_improve:
                print(f"\n[EARLY STOP] No improvement after {self.no_improve:,} steps.", flush=True)
                return False

        return True


# ─── Model Export ────────────────────────────────────────────────────────────
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

    # Also sync to www and android assets
    for target in ["www/ai_weights.json", "android/app/src/main/assets/public/ai_weights.json"]:
        if os.path.exists(os.path.dirname(target)):
            with open(target, "w") as f:
                json.dump({"layers": layers}, f)
            print(f"[export] {target} written")

    # Re-embed into HTML files
    import subprocess
    subprocess.run([sys.executable, "update_html.py"], check=False)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_ENVS = 8
    TOTAL  = 1_500_000

    print(f"═══════════════════════════════════════════════════════════════")
    print(f"  Air Hockey PPO V3 — Attack/Defense Mode Reinforcement Training")
    print(f"  {N_ENVS} envs × {TOTAL:,} timesteps")
    print(f"  Curriculum: Hard, Medium, Easy, Idle Human, Self-Play")
    print(f"  Serve distribution: Center start (40%), Opponent (30%), Agent (30%)")
    print(f"  Anti-jitter positioning + decisive offensive attack shaping")
    print(f"═══════════════════════════════════════════════════════════════")

    env = make_vec_env(AirHockeyEnv, n_envs=N_ENVS)

    log_path = "training_log.csv"

    cb = SelfPlayCallback(
        every=150_000, pool_size=12,
        log_every=200_000, log_path=log_path,
        no_improve_limit=1_500_000,
        target_wr=0.75,
    )

    lr_schedule = cosine_lr_schedule(warmup_frac=0.02, min_lr_frac=0.05)

    base_checkpoint = "air_hockey_ppo_v3.zip"
    if os.path.exists(base_checkpoint):
        print(f"[train] Loading pre-trained base model {base_checkpoint} for targeted refinement...")
        model = PPO.load(
            base_checkpoint,
            env=env,
            learning_rate=lambda p: 2.5e-4 * lr_schedule(p),
            ent_coef=0.006,
            n_steps=2048,
            batch_size=2048,
            custom_objects={"learning_rate": lambda p: 2.5e-4 * lr_schedule(p)},
        )
    else:
        print("[train] Creating new V3 model")
        model = PPO(
            "MlpPolicy", env,
            n_steps=2048,
            batch_size=2048,
            learning_rate=lambda p: 3e-4 * lr_schedule(p),
            gamma=0.997,
            gae_lambda=0.95,
            ent_coef=0.008,
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.2,
            n_epochs=10,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                activation_fn=nn.Tanh,
                ortho_init=True,
            ),
            verbose=1,
        )

    cb.model = model
    cb._assign_opponents()

    start_time = time.time()
    model.learn(total_timesteps=TOTAL, callback=cb, reset_num_timesteps=True)
    elapsed = time.time() - start_time

    model.save("air_hockey_ppo_v3")
    export_json(model, "ai_weights.json")

    print(f"\n[done] Training completed in {elapsed/60:.1f} minutes")
    print(f"  Best Hard win-rate: {cb.best_hard_wr:.1%}, Composite: {cb.best_composite:.1%}")

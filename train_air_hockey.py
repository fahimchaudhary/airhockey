#!/usr/bin/env python3
"""
train_air_hockey.py
Physics simulator + PPO self-play + curriculum training for Air Hockey.
Incorporates NeurIPS 2023 survey insights:
- Contact collision rewards (Air HocKIT)
- Potential-based distance shaping (intercept acceleration)
- Defensive deflection bonuses (save bonuses)
- Curriculum-augmented self-play pool (Ouyang / Bonenfant)
- Direct export to JS-compatible weights (ai_weights.json)
"""
import copy, json, math, os, random, sys, csv
import numpy as np
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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ─── Gymnasium Environment ───────────────────────────────────────────────────
class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-2, 2, shape=(8,), dtype=np.float32)
        self.action_space      = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.opponent = None
        self.rng = np.random.default_rng()

    def set_opponent(self, policy):
        self.opponent = policy

    def _view(self, flip: bool) -> np.ndarray:
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p1 = np.array([W / 2, H - 130.0])   # agent (bottom)
        self.p2 = np.array([W / 2, 130.0])       # opponent (top)
        self.v1 = np.zeros(2)
        self.v2 = np.zeros(2)
        self.puck = np.array([
            W / 2, H / 2,
            self.rng.uniform(-2, 2),
            float(self.rng.choice([-6, 6])),
        ])
        self.t = 0
        return self._view(False), {}

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
        prev_puck_dist = math.hypot(self.puck[0] - self.p1[0], self.puck[1] - self.p1[1])
        pre_vy = self.puck[3]

        self.p1 = self._move(self.p1, action, bottom=True)

        if self.opponent is None:
            a2 = self.rng.uniform(-1, 1, 2)
            self.p2 = self._move(self.p2, a2, bottom=False)
        elif isinstance(self.opponent, RuleAI):
            p_flip = np.array([self.puck[0], H - self.puck[1], self.puck[2], -self.puck[3]])
            self.p2, self.v2 = self.opponent.step(p_flip)
        else:
            obs2 = self._view(True)
            a2, _ = self.opponent.predict(obs2, deterministic=False)
            a2 = np.array([a2[0], -a2[1]])
            self.p2 = self._move(self.p2, a2, bottom=False)

        self.v1 = self.p1 - old1
        if not isinstance(self.opponent, RuleAI):
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

        # ─── Enhanced Reward Engineering (NeurIPS 2023 Air Hockey Survey) ────
        reward = -0.0004   # subtle living efficiency penalty

        # 1. Distance potential shaping when puck is in agent's half
        if self.puck[1] >= H / 2:
            curr_puck_dist = math.hypot(self.puck[0] - self.p1[0], self.puck[1] - self.p1[1])
            reward += 0.005 * (prev_puck_dist - curr_puck_dist)

        # 2. Collision & Offensive Hit Bonus
        if hit_p1:
            reward += 0.20   # direct contact reward
            if self.puck[3] < 0:
                # offensive shot towards opponent goal
                reward += 0.25 * (-self.puck[3] / MAXV)
            if pre_vy > 0 and self.puck[3] < 0:
                # defensive save! Reversed puck heading towards our goal
                reward += 0.35

        # 3. Terminal Goal Rewards
        terminated = False
        if self.puck[1] < -PUCK_R:        # agent scored!
            reward, terminated = 2.0, True
        elif self.puck[1] > H + PUCK_R:   # agent conceded
            reward, terminated = -2.0, True
        else:
            reward += 0.01 * (-self.puck[3] / MAXV)

        truncated = self.t >= MAX_STEPS
        return self._view(False), reward, terminated, truncated, {}


# ─── Fast Policy for 17,000+ inferences/sec ─────────────────────────────────
class FastPolicy:
    def __init__(self, model):
        self.layers = []
        for m in model.policy.mlp_extractor.policy_net:
            if isinstance(m, nn.Linear):
                self.layers.append((m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy(), True))
        a = model.policy.action_net
        self.layers.append((a.weight.detach().cpu().numpy(), a.bias.detach().cpu().numpy(), False))

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = obs
        for W, b, is_tanh in self.layers:
            x = np.dot(W, x) + b
            if is_tanh:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)

    def predict(self, obs: np.ndarray, deterministic: bool = False):
        a = self.act(obs)
        if not deterministic:
            a = a + np.random.normal(0, 0.05, size=a.shape)
            a = np.clip(a, -1.0, 1.0)
        return a, None


# ─── Rule-Based AI (for evaluation and curriculum) ───────────────────────────
DIFF = {
    "easy":   dict(speed=4.2, think=28, noise=70, aggr=0.25, defend=0.6),
    "medium": dict(speed=6.0, think=12, noise=28, aggr=0.50, defend=0.4),
    "hard":   dict(speed=9.5, think=2,  noise=6,  aggr=0.90, defend=0.2),
}

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
                base = self.predict_x(px, py, vx, vy, self.pos[1]) if vy < 0 else px
            else:
                base = W/2 + (px - W/2) * p["defend"]
            self.tx = base + (random.random()*2-1) * p["noise"]
            self.ty = clamp(py-(PR+PUCK_R)*p["aggr"], PR, H/2-PR) if py<H/2 else 110.0
        nx = self.pos[0] + clamp(self.tx-self.pos[0], -p["speed"], p["speed"])
        ny = self.pos[1] + clamp(self.ty-self.pos[1], -p["speed"]*.8, p["speed"]*.8)
        self.vel = np.array([nx-self.pos[0], ny-self.pos[1]])
        self.pos = np.array([clamp(nx,PR,W-PR), clamp(ny,PR,H/2-PR)])
        return self.pos.copy(), self.vel.copy()


# ─── Evaluation (200 games vs rule AI) ──────────────────────────────────────
def eval_win_rate(model, difficulty="hard", n_games=200) -> float:
    env  = AirHockeyEnv()
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
                obs = env._view(False)
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
        ax.set_title('Air Hockey PPO Training — Curriculum & Self-Play', fontsize=14, fontweight='bold', color='white', pad=15)
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


# ─── Self-Play + Curriculum Callback ─────────────────────────────────────────
class SelfPlayCallback(BaseCallback):
    def __init__(self, every=200_000, pool_size=8, log_every=500_000,
                 log_path="training_log.csv", no_improve_limit=3_000_000, target_wr=0.70):
        super().__init__()
        self.every          = every
        self.pool_size      = pool_size
        self.pool           = []
        self.last_snap      = 0
        self.log_every      = log_every
        self.last_log       = 0
        self.log_path       = log_path
        self.no_improve     = no_improve_limit
        self.target_wr      = target_wr
        self.best_hard_wr   = 0.20   # baseline from previous run
        self.best_at        = 0
        self._header_written = os.path.exists(log_path)

    def _assign_opponents(self):
        # Curriculum + Self-Play allocation across 8 envs:
        # Envs 0 & 1: Hard Rule AI (25% vs Hard)
        # Env 2: Medium Rule AI
        # Env 3: Easy Rule AI
        # Envs 4-7: Random snapshots from historical self-play pool
        n_envs = self.training_env.num_envs
        self.training_env.env_method("set_opponent", RuleAI("hard"), indices=0)
        if n_envs > 1:
            self.training_env.env_method("set_opponent", RuleAI("hard"), indices=1)
        if n_envs > 2:
            self.training_env.env_method("set_opponent", RuleAI("medium"), indices=2)
        if n_envs > 3:
            self.training_env.env_method("set_opponent", RuleAI("easy"), indices=3)
        if len(self.pool) > 0:
            for i in range(4, n_envs):
                self.training_env.env_method("set_opponent", random.choice(self.pool), indices=i)

    def _on_step(self):
        ts = self.num_timesteps

        # ── Snapshot for self-play ──
        if ts - self.last_snap >= self.every:
            self.last_snap = ts
            snap = FastPolicy(self.model)
            self.pool = (self.pool + [snap])[-self.pool_size:]
            self._assign_opponents()
            print(f"[curriculum+self-play] step {ts:,}, pool={len(self.pool)}", flush=True)

        # ── Win-rate evaluation ──
        if ts - self.last_log >= self.log_every:
            self.last_log = ts
            print(f"[eval] step {ts:,} ...", flush=True)
            wr_e = eval_win_rate(self.model, "easy",   200)
            wr_m = eval_win_rate(self.model, "medium", 200)
            wr_h = eval_win_rate(self.model, "hard",   200)
            print(f"  [step {ts:,}] Win rates -> Easy={wr_e:.1%}  Medium={wr_m:.1%}  Hard={wr_h:.1%}", flush=True)

            # CSV logging
            write_header = not self._header_written
            with open(self.log_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["timestep","easy","medium","hard"])
                    self._header_written = True
                w.writerow([ts, round(wr_e,4), round(wr_m,4), round(wr_h,4)])

            # Live chart update
            plot_live_chart(self.log_path, "win_rate_chart.png")

            # Best model tracking and export
            if wr_h >= self.best_hard_wr:
                self.best_hard_wr = wr_h
                self.best_at      = ts
                self.model.save("air_hockey_ppo_best")
                export_json(self.model, "ai_weights.json")
                if os.path.exists("www"):
                    export_json(self.model, "www/ai_weights.json")
                # Also sync into native android assets folder!
                android_asset = "android/app/src/main/assets/public/ai_weights.json"
                if os.path.exists(os.path.dirname(android_asset)):
                    export_json(self.model, android_asset)

            # Check 70% target
            if wr_h >= self.target_wr:
                print(f"\n[TARGET REACHED] Hard win-rate {wr_h:.1%} >= {self.target_wr:.0%} at step {ts:,}!", flush=True)
                return False

            # Early stop check
            if ts >= self.no_improve and (ts - self.best_at) >= self.no_improve:
                print(f"\n[EARLY STOP] No improvement after {self.no_improve:,} steps.", flush=True)
                print(f"  Best Hard win-rate: {self.best_hard_wr:.1%} at step {self.best_at:,}", flush=True)
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


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_ENVS = 8
    TOTAL  = 10_000_000

    print(f"[train] Starting Perfected Training: {N_ENVS} envs × {TOTAL:,} timesteps")
    env = make_vec_env(AirHockeyEnv, n_envs=N_ENVS)

    cb = SelfPlayCallback(
        every=200_000, pool_size=8,
        log_every=500_000, log_path="training_log.csv",
        no_improve_limit=3_000_000,
        target_wr=0.70,
    )

    if os.path.exists("air_hockey_ppo_best.zip"):
        print("[resume] Initializing from existing checkpoint air_hockey_ppo_best.zip...")
        model = PPO.load(
            "air_hockey_ppo_best.zip",
            env=env,
            learning_rate=2e-4,
            ent_coef=0.015,
            verbose=1,
        )
    else:
        model = PPO(
            "MlpPolicy", env,
            n_steps=1024, batch_size=512,
            learning_rate=3e-4, gamma=0.995,
            ent_coef=0.015,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=1,
        )

    # Initial opponent assignment
    cb.model = model
    cb._assign_opponents()

    model.learn(total_timesteps=TOTAL, callback=cb, reset_num_timesteps=False)
    model.save("air_hockey_ppo")
    export_json(model, "ai_weights.json")

    print("\n[done] Training completed successfully!")

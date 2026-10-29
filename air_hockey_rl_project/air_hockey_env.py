"""Lightweight Gymnasium Air Hockey environment for continuous-control RL."""
from __future__ import annotations
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

W, H = 1.0, 1.6
GOAL_HALF = 0.19
PUCK_R = 0.025
PADDLE_R = 0.065
MAX_PUCK_SPEED = 1.45
PADDLE_SPEED = 0.85
DT = 1.0 / 60.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def norm_pos(x, y):
    return np.array([2*x/W - 1, 2*y/H - 1], dtype=np.float32)

class AirHockeyEnv(gym.Env):
    """Agent controls the bottom paddle; a simple opponent controls the top paddle."""
    metadata = {"render_modes": []}

    def __init__(self, opponent_level="medium", max_steps=2400, randomize=True):
        super().__init__()
        self.opponent_level = opponent_level
        self.max_steps = max_steps
        self.randomize = randomize
        # puck(x,y,vx,vy), agent(x,y), opponent(x,y), score difference
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.steps = 0
        self.reset()

    def _obs(self):
        p = norm_pos(self.puck[0], self.puck[1])
        pv = np.array([self.puck[2]/MAX_PUCK_SPEED, self.puck[3]/MAX_PUCK_SPEED], dtype=np.float32)
        a = norm_pos(self.agent[0], self.agent[1])
        o = norm_pos(self.opp[0], self.opp[1])
        score = np.array([np.clip((self.agent_score-self.opp_score)/5, -1, 1)], dtype=np.float32)
        return np.clip(np.concatenate([p, pv, a, o, score]), -1, 1).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.agent_score = 0
        self.opp_score = 0
        self.puck = np.array([0.5, 0.8, self.rng.uniform(-0.25,0.25), self.rng.choice([-0.55,0.55])], dtype=float)
        self.agent = np.array([0.5, 1.35], dtype=float)
        self.opp = np.array([0.5, 0.25], dtype=float)
        self.prev_dist = math.hypot(self.puck[0]-self.agent[0], self.puck[1]-self.agent[1])
        return self._obs(), {}

    def _move_paddle(self, paddle, action, bottom):
        vx, vy = float(action[0])*PADDLE_SPEED, float(action[1])*PADDLE_SPEED
        paddle[0] = clamp(paddle[0] + vx*DT, PADDLE_R, W-PADDLE_R)
        if bottom:
            paddle[1] = clamp(paddle[1] + vy*DT, H*0.52, H-PADDLE_R)
        else:
            paddle[1] = clamp(paddle[1] + vy*DT, PADDLE_R, H*0.48)

    def _opponent_action(self):
        # Intercept projected puck location when it travels toward the top.
        x, y, vx, vy = self.puck
        if vy < 0:
            t = max(0, (self.opp[1] - y) / (vy if abs(vy)>1e-6 else -1e-6))
            target_x = x + vx*t
            # reflect prediction off side walls
            while target_x < 0 or target_x > W:
                if target_x < 0: target_x = -target_x
                if target_x > W: target_x = 2*W-target_x
        else:
            target_x = 0.5 + (x-0.5)*0.25
        target_y = 0.24 if vy < 0 else 0.31
        delta = np.array([target_x, target_y])-self.opp
        norm = np.linalg.norm(delta) + 1e-8
        strength = {"easy":0.45, "medium":0.75, "hard":1.0}.get(self.opponent_level, 0.75)
        return np.clip(delta/norm * strength, -1, 1)

    def _collide_paddle(self, paddle):
        dx, dy = self.puck[0]-paddle[0], self.puck[1]-paddle[1]
        dist = math.hypot(dx,dy)
        min_dist = PUCK_R+PADDLE_R
        if dist < min_dist:
            if dist < 1e-8:
                dx, dy, dist = 0.0, 1.0, 1.0
            nx, ny = dx/dist, dy/dist
            # Separate bodies to prevent sticking.
            self.puck[0] = paddle[0] + nx*(min_dist+1e-4)
            self.puck[1] = paddle[1] + ny*(min_dist+1e-4)
            dot = self.puck[2]*nx + self.puck[3]*ny
            if dot < 0:
                self.puck[2] -= 1.85*dot*nx
                self.puck[3] -= 1.85*dot*ny
            # Paddle movement transfers a little momentum to the puck.
            self.puck[2] += 0.10
            speed = math.hypot(self.puck[2], self.puck[3])
            if speed > MAX_PUCK_SPEED:
                self.puck[2:4] *= MAX_PUCK_SPEED/speed

    def step(self, action):
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        self._move_paddle(self.agent, action, True)
        self._move_paddle(self.opp, self._opponent_action(), False)

        # Substeps reduce tunnelling at high puck speed.
        reward = -0.001
        sub_dt = DT/3
        for _ in range(3):
            self.puck[0] += self.puck[2]*sub_dt
            self.puck[1] += self.puck[3]*sub_dt
            # Side rails.
            if self.puck[0] < PUCK_R:
                self.puck[0] = PUCK_R
                self.puck[2] = abs(self.puck[2])*0.97
            elif self.puck[0] > W-PUCK_R:
                self.puck[0] = W-PUCK_R
                self.puck[2] = -abs(self.puck[2])*0.97
            # Top and bottom: goals exist only through the goal mouth.
            in_goal = abs(self.puck[0]-0.5) <= GOAL_HALF
            if self.puck[1] < PUCK_R:
                if in_goal:
                    self.agent_score += 1
                    reward += 1.0
                    return self._obs(), reward, True, False, {"goal": "agent", "agent_score": self.agent_score, "opp_score": self.opp_score}
                self.puck[1] = PUCK_R
                self.puck[3] = abs(self.puck[3])*0.97
            elif self.puck[1] > H-PUCK_R:
                if in_goal:
                    self.opp_score += 1
                    reward -= 1.0
                    return self._obs(), reward, True, False, {"goal": "opponent", "agent_score": self.agent_score, "opp_score": self.opp_score}
                self.puck[1] = H-PUCK_R
                self.puck[3] = -abs(self.puck[3])*0.97
            self._collide_paddle(self.agent)
            self._collide_paddle(self.opp)

        # Mild shaping toward puck; terminal goal reward remains dominant.
        dist = math.hypot(self.puck[0]-self.agent[0], self.puck[1]-self.agent[1])
        reward += 0.008 * np.clip(self.prev_dist-dist, -0.05, 0.05)
        self.prev_dist = dist
        terminated = False
        truncated = self.steps >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {
            "agent_score": self.agent_score, "opp_score": self.opp_score
        }

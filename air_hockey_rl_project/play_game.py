"""Portrait Air Hockey. Human controls bottom paddle; AI controls top paddle."""
import argparse, math, os, sys
import pygame
from air_hockey_env import AirHockeyEnv, W, H, GOAL_HALF, PUCK_R, PADDLE_R, DT
try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None

WIDTH, HEIGHT = 480, 800
FPS = 60
BG = (9, 24, 38)
TABLE = (12, 49, 63)
LINE = (117, 204, 211)
WHITE = (244, 249, 250)
CYAN = (49, 220, 222)
RED = (255, 104, 116)
MUTED = (153, 184, 193)

def screen_pos(x, y):
    return int(x*WIDTH), int(y/H*HEIGHT)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", choices=["rule","ppo"], default="rule")
    parser.add_argument("--difficulty", choices=["easy","medium","hard"], default="medium")
    parser.add_argument("--model", default="models/air_hockey_ppo.zip")
    args = parser.parse_args()
    if args.ai == "ppo":
        if PPO is None or not os.path.exists(args.model):
            print("PPO model not found. Train it first or use --ai rule.")
            sys.exit(1)
        model = PPO.load(args.model)
    else:
        model = None

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Air Hockey RL Lab")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24, bold=True)
    small = pygame.font.SysFont("arial", 15)
    env = AirHockeyEnv(opponent_level=args.difficulty)
    env.reset()
    # Game state is shared with the environment physics.
    human = env.agent
    puck = env.puck
    opp = env.opp
    human_score = opp_score = 0
    current_diff = args.difficulty
    running = True
    key_vec = [0,0]
    last_goal = ""
    while running:
        dt = min(clock.tick(FPS)/1000.0, 0.03)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_r:
                    env.reset(); human_score = opp_score = 0; last_goal = ""
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    current_diff = {pygame.K_1:"easy",pygame.K_2:"medium",pygame.K_3:"hard"}[event.key]
                    env.opponent_level = current_diff
        keys = pygame.key.get_pressed()
        dx = (1 if keys[pygame.K_RIGHT] or keys[pygame.K_d] else 0) - (1 if keys[pygame.K_LEFT] or keys[pygame.K_a] else 0)
        dy = (1 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0) - (1 if keys[pygame.K_UP] or keys[pygame.K_w] else 0)
        if dx == 0 and dy == 0:
            mx,my = pygame.mouse.get_pos()
            target = [mx/WIDTH, my/HEIGHT*H]
            target[0] = clamp(target[0], PADDLE_R, W-PADDLE_R)
            target[1] = clamp(target[1], H*0.52, H-PADDLE_R)
            delta = [target[0]-human[0], target[1]-human[1]]
            mag = math.hypot(*delta)
            if mag > 1e-6:
                dx,dy = delta[0]/mag,delta[1]/mag
        mag = math.hypot(dx,dy)
        if mag > 1: dx,dy = dx/mag,dy/mag
        action = [dx,dy]
        if model is not None:
            obs = env._obs()
            ai_action, _ = model.predict(obs, deterministic=True)
            # Model controls the bottom paddle; invert sides for top AI by mirrored observation is not trained here.
            # For a usable default, the trained bottom policy is mirrored in action space.
            top_action = [-float(ai_action[0]), -float(ai_action[1])]
            env._move_paddle(env.agent, action, True)
            env._move_paddle(env.opp, top_action, False)
            # Advance puck/collisions through env.step while preventing a second paddle move.
            saved = env._move_paddle
            env._move_paddle = lambda paddle, act, bottom: None
            _, reward, term, trunc, info = env.step([0,0])
            env._move_paddle = saved
        else:
            _, reward, term, trunc, info = env.step(action)
        if term:
            last_goal = "YOU SCORED!" if info.get("goal") == "agent" else "AI SCORED!"
            if info.get("goal") == "agent": human_score += 1
            else: opp_score += 1
            env.reset()
        # Refresh references after reset(), which replaces the state arrays.
        human, puck, opp = env.agent, env.puck, env.opp
        # Draw portrait table.
        screen.fill(BG)
        pygame.draw.rect(screen, TABLE, pygame.Rect(18, 18, WIDTH-36, HEIGHT-36), border_radius=22)
        # Goal mouths and center line.
        gx = WIDTH//2
        gw = int(GOAL_HALF*2*WIDTH)
        pygame.draw.rect(screen, BG, (gx-gw//2, 17, gw, 16), border_radius=5)
        pygame.draw.rect(screen, BG, (gx-gw//2, HEIGHT-33, gw, 16), border_radius=5)
        pygame.draw.line(screen, LINE, (20, HEIGHT//2), (WIDTH-20, HEIGHT//2), 2)
        pygame.draw.circle(screen, LINE, (gx, HEIGHT//2), 58, 2)
        pygame.draw.circle(screen, LINE, (gx, HEIGHT//2), 5)
        # Goal indicator.
        pygame.draw.rect(screen, RED, (gx-gw//2, 18, gw, 4), border_radius=2)
        pygame.draw.rect(screen, CYAN, (gx-gw//2, HEIGHT-22, gw, 4), border_radius=2)
        # Puck and paddles.
        px,py = screen_pos(puck[0], puck[1])
        ox,oy = screen_pos(opp[0], opp[1])
        hx,hy = screen_pos(human[0], human[1])
        pygame.draw.circle(screen, (0,0,0), (px+3,py+4), int(PUCK_R*WIDTH)+2)
        pygame.draw.circle(screen, WHITE, (px,py), int(PUCK_R*WIDTH))
        pygame.draw.circle(screen, RED, (ox,oy), int(PADDLE_R*WIDTH))
        pygame.draw.circle(screen, (255,192,198), (ox,oy), int(PADDLE_R*WIDTH)-8, 3)
        pygame.draw.circle(screen, CYAN, (hx,hy), int(PADDLE_R*WIDTH))
        pygame.draw.circle(screen, (183,255,255), (hx,hy), int(PADDLE_R*WIDTH)-8, 3)
        # UI.
        score = font.render(f"{opp_score}     {human_score}", True, WHITE)
        screen.blit(score, score.get_rect(center=(WIDTH//2, 58)))
        screen.blit(small.render(f"AI: {args.ai.upper()} / {current_diff.upper()}", True, MUTED), (30, 82))
        screen.blit(small.render("MOVE: MOUSE OR WASD / ARROWS", True, MUTED), (30, HEIGHT-61))
        screen.blit(small.render("1 EASY   2 MEDIUM   3 HARD    R RESET    ESC QUIT", True, MUTED), (30, HEIGHT-40))
        if last_goal:
            msg = font.render(last_goal, True, WHITE)
            screen.blit(msg, msg.get_rect(center=(WIDTH//2, HEIGHT//2)))
        pygame.display.flip()
    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()

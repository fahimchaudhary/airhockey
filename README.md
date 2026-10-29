# 🏒 Air Hockey with Self-Learning PPO AI (Android APK + Web)

A high-performance, neon-arcade Air Hockey mobile game for Android featuring a custom physics engine and a **self-learning Deep Reinforcement Learning AI (PPO)** trained via competitive self-play.

---

## 📱 Features

- **Neon Cyberpunk Aesthetic**: High-contrast dark table (`#030712`) with glowing cyan/red boundaries, neon mallets, dynamic speed trail glows on the puck, and digital scoreboard.
- **Fixed 60 FPS Physics Engine**: Bit-exact deterministic simulation across both Python and JavaScript engines (mallet restitution 1.05, wall restitution 0.9, friction 0.995).
- **4 Difficulties**:
  - 🟢 **Easy**: Beginner AI with reaction delay and spatial noise.
  - 🟠 **Medium**: Skilled AI with defensive shading.
  - 🔴 **Hard**: Expert AI with geometric wall-bounce trajectory prediction.
  - 🤖 **Pro**: Deep Neural Network policy (8 inputs → 128 → 128 → 2 outputs) trained with PPO self-play, running real-time browser/Android inference via a lightweight ~30-line forward pass.
- **Web Audio & Haptics**: Real-time synthesized impact clicks, low-frequency wall thuds, goal fanfare chimes, and tactile vibration feedback (`navigator.vibrate`) with HUD mute toggle.
- **Dynamic Particles & Motion Trail**: Glowing neon spark bursts on collisions, victory confetti explosions, and high-speed motion blur trails for the puck.
- **Android Native App**: Fullscreen, immersive sticky mode (status bar and navigation bar hidden), locked to portrait mode, zero screen scrolling, and screen wake lock.

---

## 📦 Project Deliverables

| File / Folder | Description |
|---|---|
| [`app-debug.apk`](app-debug.apk) | Ready-to-install Android Debug APK (4.6 MB) with Audio + Haptics |
| [`app-release.aab`](app-release.aab) | Android App Bundle for Google Play Release (3.3 MB) |
| [`Air Hockey.html`](Air%20Hockey.html) / [`air-hockey.html`](air-hockey.html) | Complete single-file game with Pro mode and canvas UI |
| [`train_air_hockey.py`](train_air_hockey.py) | Gymnasium environment, PPO self-play training loop, and weight exporter |
| [`eval_win_rate.py`](eval_win_rate.py) | 200-game evaluation benchmark against rule-based AIs & physics validator |
| [`ai_weights.json`](ai_weights.json) | Exported MLP weights for client-side JavaScript execution |
| [`training_log.csv`](training_log.csv) | Win-rate logs recorded every 500k timesteps |
| [`win_rate_chart.png`](win_rate_chart.png) | High-resolution visual win rate progression chart |
| [`android/`](android/) | Capacitor Android Studio project folder |

---

## 📲 How to Install & Play on Android

### 1. Direct ADB Install
Connect your Android phone with USB debugging enabled, then run:
```bash
adb install -r app-debug.apk
```

### 2. Manual Installation
Copy `app-debug.apk` to your phone's storage, open your file manager, and tap to install (enable "Install unknown apps" if prompted).

---

## 🎮 How to Play
1. Launch the app or open `Air Hockey.html` in any browser.
2. Select your desired difficulty: **Easy**, **Medium**, **Hard**, or **Pro (🤖 AI)**.
3. First to **7 points** wins.
4. Drag your finger or mouse across the bottom half of the table to swing your mallet.

---

## 🔬 Reinforcement Learning Details

- **Algorithm**: Proximal Policy Optimization (PPO) via Stable-Baselines3.
- **Self-Play Mechanism**: Every 200,000 steps, a snapshot of the current policy is frozen and added to a historical opponent pool (size 8). Training environments randomly sample opponents from this pool to prevent policy collapse and catastrophic forgetting.
- **Observation Space (8 floats, normalized `[-1, 1]`)**:
  - `[px, py, vx, vy, mallet_x, mallet_y, opponent_x, opponent_y]`
- **Action Space (2 floats, continuous `[-1, 1]`)**:
  - `[dx, dy]` normalized velocity vectors (multiplied by `MALLET_SPEED = 12.0`).
- **Reward Function**:
  - `+1.0` for scoring a goal.
  - `-1.0` for conceding a goal.
  - `+0.01 * (-puck_vy / MAXV)` shaping reward for keeping the puck on the opponent's half.
- **Early Stop Safeguard**: Automatic termination if no win-rate improvement occurs over 3,000,000 steps, or early completion upon reaching 70% win-rate against Hard AI.

---

## 🛠️ How to Rebuild Everything from Scratch

### 1. Python Environment Setup & Training
```bash
# Setup virtual environment
uv venv .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install "stable-baselines3>=2.0.0" gymnasium numpy matplotlib

# Train PPO self-play agent
python -u train_air_hockey.py

# Evaluate agent and plot chart
python eval_win_rate.py air_hockey_ppo_best
```

### 2. Android APK & AAB Rebuild
```bash
# Sync web assets to native Android project
npx cap sync android

# Build Debug APK
cd android
./gradlew assembleDebug
# Generated at: android/app/build/outputs/apk/debug/app-debug.apk

# Build Release AAB
./gradlew bundleRelease
# Generated at: android/app/build/outputs/bundle/release/app-release.aab
```

---

## ⚡ Physics Verification
The simulator maintains bit-exact fidelity with the browser implementation. Verified over 1000 random trajectory steps:
```
Maximum difference across 1000 steps: 0.00e+00
Status: BIT-EXACT MATCH (0.0 error)
```

import json

with open('/home/fahim/game/ai_weights.json') as f:
    weights_json = f.read().strip()

template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">
<meta name="theme-color" content="#030712">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Air Hockey</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root {
  --c-bg:     #030712;
  --c-red:    #ff3a5c;
  --c-blue:   #00c8ff;
  --c-purple: #a855f7;
  --safe-top: env(safe-area-inset-top,0px);
  --safe-bot: env(safe-area-inset-bottom,0px);
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html,body{
  height:100%;width:100%;overflow:hidden;
  background:var(--c-bg);
  font-family:'Inter',system-ui,sans-serif;
  color:#fff;user-select:none;-webkit-user-select:none;
}
#wrap{
  position:fixed;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  padding:calc(var(--safe-top) + 8px) 8px calc(var(--safe-bot) + 8px);
}
canvas{touch-action:none;display:block;border-radius:20px;
  box-shadow:0 0 0 1px rgba(255,255,255,.08),0 0 60px rgba(0,180,255,.18),0 0 120px rgba(0,80,200,.12);}
#hud{width:100%;max-width:400px;display:none;align-items:center;
  justify-content:space-between;padding:0 4px 8px;gap:8px;}
#hud-ai,#hud-you{display:flex;flex-direction:column;align-items:center;gap:2px;}
#hud-ai .label{color:var(--c-blue);}
#hud-you .label{color:var(--c-red);}
.label{font-family:'Orbitron',monospace;font-size:9px;letter-spacing:2px;opacity:.7}
.score{font-family:'Orbitron',monospace;font-size:28px;font-weight:900;line-height:1}
#score-ai{color:var(--c-blue);text-shadow:0 0 20px var(--c-blue);}
#score-you{color:var(--c-red);text-shadow:0 0 20px var(--c-red);}
#hud-mid{display:flex;flex-direction:column;align-items:center;gap:4px}
#mode-badge{font-family:'Orbitron',monospace;font-size:8px;letter-spacing:2px;
  padding:3px 10px;border-radius:20px;background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.12);opacity:.8;}
.hud-btns{display:flex;gap:6px;align-items:center;}
#pause-btn, #sound-btn{
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);
  border-radius:10px;padding:4px 10px;font-size:12px;font-weight:600;
  color:rgba(255,255,255,.7);cursor:pointer;transition:all .15s;
}
#pause-btn:active, #sound-btn:active{background:rgba(255,255,255,.18);transform:scale(.96)}
#menu{position:fixed;inset:0;z-index:10;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:0;
  padding:calc(var(--safe-top) + 20px) 24px calc(var(--safe-bot) + 20px);
  background:radial-gradient(ellipse at 50% 20%,rgba(0,100,255,.15) 0%,transparent 60%),
             radial-gradient(ellipse at 50% 80%,rgba(255,30,80,.12) 0%,transparent 60%),
             rgba(3,7,18,.94);
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);}
.menu-logo{position:relative;z-index:1;margin-bottom:6px;text-align:center;}
.menu-logo h1{font-family:'Orbitron',monospace;font-size:clamp(28px,9vw,50px);font-weight:900;
  letter-spacing:4px;background:linear-gradient(135deg,#fff 0%,var(--c-blue) 50%,var(--c-red) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  filter:drop-shadow(0 0 20px rgba(0,180,255,.4));line-height:1;}
.menu-logo p{font-size:11px;letter-spacing:3px;text-transform:uppercase;
  color:rgba(255,255,255,.4);text-align:center;margin-top:6px;}
.puck-icon{width:44px;height:44px;border-radius:50%;
  background:radial-gradient(circle at 35% 35%,#555,#111);
  box-shadow:0 0 0 2px rgba(255,255,255,.1),0 8px 20px rgba(0,0,0,.6),0 0 30px rgba(100,180,255,.15);
  margin:16px auto 20px;position:relative;}
.puck-icon::after{content:'';position:absolute;top:50%;left:50%;
  transform:translate(-50%,-50%);width:20px;height:20px;border-radius:50%;
  border:2px solid rgba(140,210,255,.5);}
.menu-sub{font-size:13px;color:rgba(255,255,255,.5);text-align:center;
  line-height:1.5;margin-bottom:24px;max-width:260px;position:relative;z-index:1;}
.btn-group{display:flex;flex-direction:column;gap:10px;width:100%;max-width:300px;position:relative;z-index:1;}
.diff-btn{position:relative;overflow:hidden;border:0;border-radius:16px;padding:14px 18px;
  font-family:'Inter',system-ui,sans-serif;font-size:15px;font-weight:800;color:#fff;cursor:pointer;
  display:flex;align-items:center;justify-content:space-between;transition:transform .1s,filter .1s;letter-spacing:.5px;}
.diff-btn::before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,rgba(255,255,255,.12) 0%,transparent 100%);}
.diff-btn:active{transform:scale(.97);filter:brightness(.9)}
.diff-btn .btn-label{display:flex;align-items:center;gap:10px}
.diff-btn .btn-icon{font-size:20px;line-height:1}
.diff-btn .btn-name{font-size:15px}
.diff-btn .btn-desc{font-size:10px;opacity:.65;font-weight:600;letter-spacing:.5px}
.diff-btn .btn-chevron{opacity:.5;font-size:18px}
.btn-easy{background:linear-gradient(135deg,#15803d,#22c55e)}
.btn-medium{background:linear-gradient(135deg,#c2410c,#f97316)}
.btn-hard{background:linear-gradient(135deg,#9f1239,#f43f5e)}
.btn-pro{background:linear-gradient(135deg,#6b21a8,#a855f7,#c084fc);box-shadow:0 0 30px rgba(168,85,247,.35);}
</style>
</head>
<body>
<div id="wrap">
  <div id="hud">
    <div id="hud-ai"><div class="label">AI</div><div class="score" id="score-ai">0</div></div>
    <div id="hud-mid">
      <div id="mode-badge">EASY</div>
      <div class="hud-btns">
        <button id="sound-btn" title="Toggle Sound">🔊</button>
        <button id="pause-btn">⏸ MENU</button>
      </div>
    </div>
    <div id="hud-you"><div class="label">YOU</div><div class="score" id="score-you">0</div></div>
  </div>
  <canvas id="c"></canvas>
</div>

<div id="menu">
  <div class="menu-logo">
    <h1 id="menu-title">AIR HOCKEY</h1>
    <p>Neon Table · Touch Control</p>
  </div>
  <div class="puck-icon"></div>
  <p class="menu-sub" id="menu-sub">First to 7 wins. Drag your mallet on the bottom half.</p>
  <div class="btn-group">
    <button class="diff-btn btn-easy"   data-mode="easy">
      <span class="btn-label"><span class="btn-icon">🟢</span><span class="btn-name">Easy</span></span>
      <span class="btn-desc">BEGINNER</span><span class="btn-chevron">›</span>
    </button>
    <button class="diff-btn btn-medium" data-mode="medium">
      <span class="btn-label"><span class="btn-icon">🟠</span><span class="btn-name">Medium</span></span>
      <span class="btn-desc">SKILLED</span><span class="btn-chevron">›</span>
    </button>
    <button class="diff-btn btn-hard"   data-mode="hard">
      <span class="btn-label"><span class="btn-icon">🔴</span><span class="btn-name">Hard</span></span>
      <span class="btn-desc">EXPERT</span><span class="btn-chevron">›</span>
    </button>
    <button class="diff-btn btn-pro"    data-mode="pro" id="btn-pro">
      <span class="btn-label"><span class="btn-icon">🤖</span><span class="btn-name">Pro</span></span>
      <span class="btn-desc">PPO NEURAL NET</span><span class="btn-chevron">›</span>
    </button>
  </div>
</div>

<script id="embedded-weights" type="application/json">
__WEIGHTS_JSON__
</script>

<script>
// ═══ CONSTANTS (sync with train_air_hockey.py) ═══
const W = 400, H = 700;
const GOAL_W = 150;
const PR = 34;
const PUCK_R = 18;
const WIN = 7;
const MAX_PUCK_SPEED = 26;
const MALLET_SPEED = 12;
const N_FRAMES = 3;  // frame stacking count

const canvas      = document.getElementById('c');
const ctx         = canvas.getContext('2d');
const menuEl      = document.getElementById('menu');
const hudEl       = document.getElementById('hud');
const menuSub     = document.getElementById('menu-sub');
const menuTitle   = document.getElementById('menu-title');
const scoreAiEl   = document.getElementById('score-ai');
const scoreYouEl  = document.getElementById('score-you');
const modeBadge   = document.getElementById('mode-badge');
const pauseBtn    = document.getElementById('pause-btn');
const soundBtn    = document.getElementById('sound-btn');

// ═══ SIZING ═══
let scale = 1, dpr = 1;
function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 3);
  const hudH = hudEl.offsetHeight || 0;
  const padV = 16, padH = 16;
  const vw = window.innerWidth  - padH * 2;
  const vh = window.innerHeight - hudH - padV * 2;
  scale = Math.min(vw / W, vh / H) * 0.98;
  canvas.width  = Math.round(W * scale * dpr);
  canvas.height = Math.round(H * scale * dpr);
  canvas.style.width  = (W * scale) + 'px';
  canvas.style.height = (H * scale) + 'px';
  hudEl.style.width = (W * scale) + 'px';
}
window.addEventListener('resize', resize);
resize();

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ═══ AUDIO & HAPTICS ═══
let audioCtx = null;
let soundEnabled = true;
function initAudio() {
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (Ctx) audioCtx = new Ctx();
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}
function playSound(type, intensity = 1) {
  if (!soundEnabled) return;
  initAudio();
  if (!audioCtx) return;
  const now = audioCtx.currentTime;
  if (type === 'hit') {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    const freq = 320 + Math.min(intensity, 20) * 22;
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(freq, now);
    osc.frequency.exponentialRampToValueAtTime(80, now + 0.08);
    gain.gain.setValueAtTime(Math.min(0.25 * intensity, 0.5), now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.start(now); osc.stop(now + 0.08);
    if (navigator.vibrate) navigator.vibrate(intensity > 12 ? 28 : 14);
  } else if (type === 'wall') {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(140, now);
    osc.frequency.exponentialRampToValueAtTime(50, now + 0.06);
    gain.gain.setValueAtTime(0.18, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.start(now); osc.stop(now + 0.06);
  } else if (type === 'goal') {
    [440, 554, 659, 880].forEach((f, i) => {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'square';
      osc.frequency.setValueAtTime(f, now + i * 0.07);
      gain.gain.setValueAtTime(0.15, now + i * 0.07);
      gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.07 + 0.22);
      osc.connect(gain); gain.connect(audioCtx.destination);
      osc.start(now + i * 0.07); osc.stop(now + i * 0.07 + 0.22);
    });
    if (navigator.vibrate) navigator.vibrate([60, 40, 100]);
  } else if (type === 'serve') {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(587, now);
    osc.frequency.exponentialRampToValueAtTime(880, now + 0.12);
    gain.gain.setValueAtTime(0.18, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.start(now); osc.stop(now + 0.12);
  }
}
soundBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  soundEnabled = !soundEnabled;
  soundBtn.textContent = soundEnabled ? '🔊' : '🔇';
  soundBtn.style.opacity = soundEnabled ? '1' : '0.5';
  initAudio();
});

// ═══ PARTICLES & TRAIL ═══
const particles = [];
function spawnParticles(x, y, color, count = 10, speed = 4) {
  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const v = (Math.random() * 0.7 + 0.3) * speed;
    particles.push({
      x, y,
      vx: Math.cos(angle) * v, vy: Math.sin(angle) * v,
      color,
      size: Math.random() * 2.5 + 1.2,
      life: 1.0,
      decay: Math.random() * 0.04 + 0.03
    });
  }
}
function updateParticles() {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx; p.y += p.vy;
    p.life -= p.decay;
    if (p.life <= 0) particles.splice(i, 1);
  }
}
function drawParticles() {
  for (const p of particles) {
    ctx.save();
    ctx.globalAlpha = Math.max(0, p.life);
    ctx.fillStyle = p.color;
    ctx.shadowColor = p.color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

const trail = [];
const MAX_TRAIL = 6;
function updateTrail() {
  const spd = Math.hypot(puck.vx, puck.vy);
  if (spd > 3) {
    trail.unshift({ x: puck.x, y: puck.y, spd });
    if (trail.length > MAX_TRAIL) trail.pop();
  } else {
    if (trail.length > 0) trail.pop();
  }
}
function drawTrail() {
  if (trail.length < 2) return;
  for (let i = 0; i < trail.length - 1; i++) {
    const alpha = (1 - i / trail.length) * 0.25;
    ctx.save();
    ctx.strokeStyle = `rgba(0, 200, 255, ${alpha})`;
    ctx.lineWidth = PUCK_R * 1.4 * (1 - i / trail.length);
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(trail[i].x, trail[i].y);
    ctx.lineTo(trail[i + 1].x, trail[i + 1].y);
    ctx.stroke();
    ctx.restore();
  }
}

// ═══ STATE ═══
let state = 'menu', mode = 'medium';
let scores = [0, 0];
let serveTimer = 0, serveDir = 1;
const puck    = { x: W/2, y: H/2, vx: 0, vy: 0 };
const human   = { x: W/2, y: H-130, vx: 0, vy: 0 };
const ai      = { x: W/2, y: 130,   vx: 0, vy: 0, timer: 0, tx: W/2, ty: 130 };
const pointer = { x: W/2, y: H-130, active: false };

// ═══ DIFFICULTY ═══
const DIFF = {
  easy:   { speed: 5.0, think: 14, noise: 25, aggr: 0.35, defend: 0.55 },
  medium: { speed: 7.2, think: 7,  noise: 10, aggr: 0.65, defend: 0.35 },
  hard:   { speed: 10.5, think: 2,  noise: 2,  aggr: 0.95, defend: 0.15 },
};

// ═══ PRO AI — MLP forward pass with frame stacking ═══
let AI_NET = null;
// Frame stack buffer for Pro AI (stores last N_FRAMES of 8-dim observations)
let proFrameStack = [];

async function loadProAI() {
  try {
    const r = await fetch('ai_weights.json');
    if (!r.ok) throw 0;
    AI_NET = await r.json();
    console.log('[Pro AI] loaded via fetch, layers:', AI_NET.layers.length);
    const pBtn = document.getElementById('btn-pro');
    if (pBtn) pBtn.style.opacity = '1';
    return;
  } catch(_) {
    console.log('[Pro AI] fetch failed, attempting embedded fallback...');
  }
  const el = document.getElementById('embedded-weights');
  if (el) {
    try {
      AI_NET = JSON.parse(el.textContent);
      console.log('[Pro AI] loaded via embedded fallback, layers:', AI_NET.layers.length);
      const pBtn = document.getElementById('btn-pro');
      if (pBtn) pBtn.style.opacity = '1';
    } catch(e) {
      console.error('[Pro AI] embedded fallback failed:', e);
    }
  }
}
loadProAI();

function netAct(obs) {
  if (!AI_NET) return null;
  let x = obs;
  for (const L of AI_NET.layers) {
    const y = new Array(L.b.length);
    for (let i = 0; i < L.b.length; i++) {
      let s = L.b[i];
      for (let j = 0; j < L.W[i].length; j++) s += L.W[i][j] * x[j];
      y[i] = L.act === 'tanh' ? Math.tanh(s) : s;
    }
    x = y;
  }
  return x.map(v => clamp(v, -1, 1));
}

/**
 * Build a single 8-dim observation frame for the Pro AI (from AI's perspective).
 * The AI is at the top, so we flip Y coordinates.
 */
function proRawObs() {
  return [
    puck.x/W*2-1, (H-puck.y)/H*2-1, puck.vx/MAX_PUCK_SPEED, -puck.vy/MAX_PUCK_SPEED,
    ai.x/W*2-1, (H-ai.y)/H*2-1, human.x/W*2-1, (H-human.y)/H*2-1,
  ];
}

/**
 * Build frame-stacked observation for the Pro AI.
 * If the network expects 24 inputs (3 frames × 8), we stack.
 * If the network expects 8 inputs (legacy), we just use the raw obs.
 */
function proStackedObs() {
  const raw = proRawObs();
  // Auto-detect: check if the first layer expects stacked input
  const expectedInputDim = AI_NET && AI_NET.layers.length > 0 ? AI_NET.layers[0].W[0].length : 8;

  if (expectedInputDim <= 8) {
    // Legacy model: no frame stacking needed
    return raw;
  }

  // Frame-stacked model
  proFrameStack.push(raw);
  while (proFrameStack.length < N_FRAMES) {
    proFrameStack.unshift(raw.slice());  // pad with copies
  }
  if (proFrameStack.length > N_FRAMES) {
    proFrameStack = proFrameStack.slice(-N_FRAMES);
  }

  // Concatenate all frames
  const stacked = [];
  for (const frame of proFrameStack) {
    for (const v of frame) stacked.push(v);
  }
  return stacked;
}

// ═══ INPUT ═══
canvas.addEventListener('pointerdown', e => { initAudio(); pointer.active=true; updatePtr(e); e.preventDefault(); });
canvas.addEventListener('pointermove', e => { if(pointer.active) updatePtr(e); e.preventDefault(); });
canvas.addEventListener('pointerup',   () => { pointer.active=false; });
canvas.addEventListener('pointercancel',()=> { pointer.active=false; });
function updatePtr(e) {
  const r = canvas.getBoundingClientRect();
  pointer.x = (e.clientX - r.left) / r.width  * W;
  pointer.y = (e.clientY - r.top)  / r.height * H;
}

// ═══ FLOW ═══
document.querySelectorAll('[data-mode]').forEach(btn =>
  btn.addEventListener('click', () => startMatch(btn.dataset.mode))
);
pauseBtn.addEventListener('click', () => showMenu('AIR HOCKEY', 'Pick a difficulty to start a new match.'));

function startMatch(m) {
  initAudio();
  mode = m; scores = [0,0]; state = 'play';
  particles.length = 0; trail.length = 0;
  proFrameStack = [];  // Reset frame stack
  resetPositions(Math.random() < 0.5 ? 1 : -1);
  menuEl.style.display = 'none';
  hudEl.style.display  = 'flex';
  modeBadge.textContent = m.toUpperCase();
  modeBadge.style.color = m==='pro'?'#c084fc':m==='hard'?'#f43f5e':m==='medium'?'#f97316':'#22c55e';
  updateHUD();
}
function showMenu(title, sub) {
  state = 'menu';
  menuTitle.textContent = title;
  menuSub.textContent = sub;
  menuEl.style.display = 'flex';
  hudEl.style.display  = 'none';
}
function resetPositions(dir) {
  puck.x=W/2; puck.y=H/2; puck.vx=0; puck.vy=0;
  serveTimer=60; serveDir=dir;
  human.x=W/2; human.y=H-130; human.vx=0; human.vy=0;
  ai.x=W/2; ai.y=130; ai.vx=0; ai.vy=0; ai.timer=0;
  pointer.x=W/2; pointer.y=H-130; pointer.active=false;
  proFrameStack = [];  // Reset frame stack on position reset
}
function updateHUD() {
  scoreAiEl.textContent  = scores[1];
  scoreYouEl.textContent = scores[0];
}
function goal(humanScored) {
  scores[humanScored?0:1]++;
  updateHUD();
  playSound('goal');
  spawnParticles(W/2, humanScored ? 12 : H-12, humanScored ? '#ff3a5c' : '#00c8ff', 35, 7);
  if (scores[0]>=WIN || scores[1]>=WIN) {
    const won = scores[0]>scores[1];
    showMenu(won?'🏆 YOU WIN!':'🤖 AI WINS', `${scores[0]} – ${scores[1]}. Rematch?`);
    return;
  }
  resetPositions(humanScored ? -1 : 1);
}

// ═══ AI ═══
function predictX(p, y) {
  if (p.vy===0) return p.x;
  const t = (y-p.y)/p.vy; if(t<0) return p.x;
  const min=PUCK_R, max=W-PUCK_R, span=max-min;
  let x=p.x+p.vx*t, u=(x-min)%(2*span);
  if(u<0) u+=2*span; if(u>span) u=2*span-u;
  return min+u;
}

function updateAI() {
  if (mode==='pro' && AI_NET) {
    // Pro AI: Deep Reinforcement Learning with temporal frame stacking
    const obs = proStackedObs();
    const a = netAct(obs);
    if (a) {
      const ox = ai.x, oy = ai.y;
      ai.x = clamp(ai.x + a[0] * MALLET_SPEED, PR, W - PR);
      ai.y = clamp(ai.y - a[1] * MALLET_SPEED, PR, H / 2 - PR);
      ai.vx = ai.x - ox; ai.vy = ai.y - oy;
    }
    return;
  }
  const p = DIFF[mode] || DIFF.hard;
  ai.timer++;
  if (ai.timer >= p.think) {
    ai.timer = 0;
    if (puck.y < H / 2) {
      // Puck is in AI's defensive/attacking half
      if (puck.y < ai.y + 10) {
        // Puck is behind the mallet near goal!
        // Anti-hallucination / self-goal protection: never hit puck toward own goal
        if (Math.abs(puck.x - ai.x) < PR + PUCK_R + 15) {
          ai.tx = puck.x > W / 2 ? puck.x - (PR + PUCK_R + 12) : puck.x + (PR + PUCK_R + 12);
        } else {
          ai.tx = W / 2 + (puck.x - W / 2) * 0.4;
        }
        ai.ty = Math.max(ai.y, 110);
      } else {
        // Puck is in front of the mallet: intercept & strike forward
        const interceptY = clamp(ai.y, PR + 10, H / 2 - PR);
        const base = puck.vy < 0 ? predictX(puck, interceptY) : puck.x;
        ai.tx = base + (Math.random() * 2 - 1) * p.noise;
        ai.ty = clamp(puck.y - (PR + PUCK_R) * p.aggr, PR, H / 2 - PR);
      }
    } else {
      // Puck is in human player's half: maintain strategic defensive formation
      const targetBase = W / 2 + (puck.x - W / 2) * p.defend;
      ai.tx = targetBase + (Math.random() * 2 - 1) * (p.noise * 0.5);
      ai.ty = 110 + (puck.y / H) * 20;
    }
  }
  const nx = ai.x + clamp(ai.tx - ai.x, -p.speed, p.speed);
  const ny = ai.y + clamp(ai.ty - ai.y, -p.speed * 0.85, p.speed * 0.85);
  ai.vx = nx - ai.x; ai.vy = ny - ai.y;
  ai.x = clamp(nx, PR, W - PR); ai.y = clamp(ny, PR, H / 2 - PR);
}

// ═══ PHYSICS ═══
function collide(p) {
  const dx=puck.x-p.x, dy=puck.y-p.y, d=Math.hypot(dx,dy), minD=PR+PUCK_R;
  if(d<minD && d>0){
    const nx=dx/d, ny=dy/d;
    puck.x=p.x+nx*minD; puck.y=p.y+ny*minD;
    const vn=(puck.vx-p.vx)*nx+(puck.vy-p.vy)*ny;
    if(vn<0){
      const j=-(1+1.05)*vn; puck.vx+=j*nx; puck.vy+=j*ny;
      const sp=Math.hypot(puck.vx,puck.vy);
      playSound('hit', sp);
      spawnParticles(puck.x, puck.y, p===human?'#ff3a5c':'#00c8ff', 8+Math.min(sp,12), 4+sp*0.2);
    }
    const sp=Math.hypot(puck.vx,puck.vy);
    if(sp>MAX_PUCK_SPEED){puck.vx*=MAX_PUCK_SPEED/sp; puck.vy*=MAX_PUCK_SPEED/sp;}
  }
}
function updatePuck() {
  puck.vx*=0.995; puck.vy*=0.995;
  puck.x+=puck.vx; puck.y+=puck.vy;
  if(puck.x<PUCK_R){
    puck.x=PUCK_R; puck.vx=-puck.vx*.9;
    playSound('wall');
    spawnParticles(puck.x, puck.y, '#78b4ff', 4, 3);
  }
  if(puck.x>W-PUCK_R){
    puck.x=W-PUCK_R; puck.vx=-puck.vx*.9;
    playSound('wall');
    spawnParticles(puck.x, puck.y, '#78b4ff', 4, 3);
  }
  const inMouth=Math.abs(puck.x-W/2)<GOAL_W/2;
  if(puck.y<PUCK_R&&!inMouth){
    puck.y=PUCK_R; puck.vy=-puck.vy*.9;
    playSound('wall');
    spawnParticles(puck.x, puck.y, '#78b4ff', 4, 3);
  }
  if(puck.y>H-PUCK_R&&!inMouth){
    puck.y=H-PUCK_R; puck.vy=-puck.vy*.9;
    playSound('wall');
    spawnParticles(puck.x, puck.y, '#78b4ff', 4, 3);
  }
  if(puck.y<-PUCK_R){goal(true);return;}
  if(puck.y>H+PUCK_R){goal(false);return;}
}
function updateHuman() {
  const tx=clamp(pointer.x,PR,W-PR), ty=clamp(pointer.y,H/2+PR,H-PR);
  const ox=human.x, oy=human.y;
  human.x+=clamp(tx-human.x,-35,35); human.y+=clamp(ty-human.y,-35,35);
  human.vx=human.x-ox; human.vy=human.y-oy;
}
function update() {
  if(state!=='play') return;
  updateHuman(); updateAI();
  if(serveTimer>0){
    serveTimer--;
    if(serveTimer===0){
      puck.vx=(Math.random()-.5)*3; puck.vy=serveDir*6;
      playSound('serve');
    }
  }
  updatePuck();
  if(state!=='play') return;
  collide(human); collide(ai);
}

// ═══ RENDERING ═══
function roundRect(x,y,w,h,r){
  ctx.beginPath();ctx.moveTo(x+r,y);
  ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();
}
function drawTable(){
  const bg=ctx.createLinearGradient(0,0,0,H);
  bg.addColorStop(0,'#0a1628'); bg.addColorStop(.5,'#091221'); bg.addColorStop(1,'#0a1628');
  roundRect(0,0,W,H,0); ctx.fillStyle=bg; ctx.fill();
  ctx.fillStyle='rgba(255,255,255,.025)';
  for(let x=25;x<W;x+=30) for(let y=25;y<H;y+=30)
    {ctx.beginPath();ctx.arc(x,y,1.5,0,Math.PI*2);ctx.fill();}
  ctx.save(); ctx.shadowColor='rgba(100,180,255,.5)'; ctx.shadowBlur=14;
  ctx.strokeStyle='rgba(120,180,255,.55)'; ctx.lineWidth=2.5;
  roundRect(7,7,W-14,H-14,14); ctx.stroke(); ctx.restore();
  ctx.save(); ctx.setLineDash([12,10]);
  ctx.shadowColor='rgba(255,255,255,.15)'; ctx.shadowBlur=8;
  ctx.strokeStyle='rgba(255,255,255,.2)'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(12,H/2); ctx.lineTo(W-12,H/2); ctx.stroke();
  ctx.setLineDash([]); ctx.restore();
  ctx.save(); ctx.shadowColor='rgba(255,255,255,.15)'; ctx.shadowBlur=10;
  ctx.strokeStyle='rgba(255,255,255,.18)'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(W/2,H/2,60,0,Math.PI*2); ctx.stroke();
  ctx.fillStyle='rgba(255,255,255,.25)';
  ctx.beginPath(); ctx.arc(W/2,H/2,4,0,Math.PI*2); ctx.fill(); ctx.restore();
  ctx.save(); ctx.strokeStyle='rgba(0,200,255,.12)'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.arc(W/2,H/2,115,Math.PI,0); ctx.stroke();
  ctx.strokeStyle='rgba(255,60,90,.12)';
  ctx.beginPath(); ctx.arc(W/2,H/2,115,0,Math.PI); ctx.stroke(); ctx.restore();
  const gx=W/2-GOAL_W/2;
  ctx.save(); ctx.shadowColor='#ff3a5c'; ctx.shadowBlur=22;
  const gTop=ctx.createLinearGradient(gx,0,gx+GOAL_W,0);
  gTop.addColorStop(0,'transparent'); gTop.addColorStop(.5,'#ff3a5c'); gTop.addColorStop(1,'transparent');
  ctx.fillStyle=gTop; ctx.fillRect(gx,0,GOAL_W,9); ctx.restore();
  ctx.save(); ctx.shadowColor='#00c8ff'; ctx.shadowBlur=22;
  const gBot=ctx.createLinearGradient(gx,0,gx+GOAL_W,0);
  gBot.addColorStop(0,'transparent'); gBot.addColorStop(.5,'#00c8ff'); gBot.addColorStop(1,'transparent');
  ctx.fillStyle=gBot; ctx.fillRect(gx,H-9,GOAL_W,9); ctx.restore();
  ctx.save(); ctx.font='900 88px Orbitron,monospace'; ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillStyle='rgba(0,200,255,.06)'; ctx.fillText(scores[1],W/2,H/2-155);
  ctx.fillStyle='rgba(255,60,90,.06)';  ctx.fillText(scores[0],W/2,H/2+155);
  ctx.restore();
}
function drawPuck(){
  ctx.save();
  ctx.shadowColor='rgba(0,0,0,.8)'; ctx.shadowBlur=14; ctx.shadowOffsetY=6;
  const g=ctx.createRadialGradient(puck.x-6,puck.y-6,2,puck.x,puck.y,PUCK_R);
  g.addColorStop(0,'#3a3a3a'); g.addColorStop(1,'#111');
  ctx.fillStyle=g; ctx.beginPath(); ctx.arc(puck.x,puck.y,PUCK_R,0,Math.PI*2); ctx.fill();
  ctx.restore();
  ctx.strokeStyle='rgba(140,210,255,.4)'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(puck.x,puck.y,PUCK_R*.52,0,Math.PI*2); ctx.stroke();
  const spd=Math.hypot(puck.vx,puck.vy);
  if(spd>3){
    ctx.save(); ctx.shadowColor='rgba(180,220,255,.5)'; ctx.shadowBlur=8+spd*2;
    ctx.strokeStyle='rgba(180,220,255,.15)'; ctx.lineWidth=3;
    ctx.beginPath(); ctx.arc(puck.x,puck.y,PUCK_R,0,Math.PI*2); ctx.stroke(); ctx.restore();
  }
}
function drawMallet(x,y,color,glow){
  ctx.save(); ctx.shadowColor=glow; ctx.shadowBlur=28;
  const g=ctx.createRadialGradient(x-PR*.35,y-PR*.35,PR*.08,x,y,PR);
  g.addColorStop(0,'#ffffff'); g.addColorStop(.28,color); g.addColorStop(.7,color+'aa'); g.addColorStop(1,'#060c1a');
  ctx.fillStyle=g; ctx.beginPath(); ctx.arc(x,y,PR,0,Math.PI*2); ctx.fill(); ctx.restore();
  ctx.strokeStyle='rgba(255,255,255,.5)'; ctx.lineWidth=2.5;
  ctx.beginPath(); ctx.arc(x,y,PR*.44,0,Math.PI*2); ctx.stroke();
  ctx.fillStyle='rgba(255,255,255,.65)';
  ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fill();
}
function drawServeRing(){
  if(serveTimer<=0) return;
  const prog=1-serveTimer/60;
  ctx.save(); ctx.strokeStyle='rgba(255,255,255,.3)'; ctx.lineWidth=3;
  ctx.beginPath(); ctx.arc(W/2,H/2,78,-Math.PI/2,-Math.PI/2+prog*Math.PI*2); ctx.stroke(); ctx.restore();
}
function draw(){
  ctx.setTransform(scale*dpr,0,0,scale*dpr,0,0);
  ctx.clearRect(0,0,W,H);
  drawTable(); drawTrail(); drawServeRing(); drawParticles(); drawPuck();
  drawMallet(ai.x,ai.y,'#00c8ff','#00c8ff');
  drawMallet(human.x,human.y,'#ff3a5c','#ff3a5c');
}

// ═══ LOOP ═══
let acc=0, last=performance.now();
function frame(now){
  const dt=Math.min(now-last,100); last=now;
  acc+=dt/(1000/60);
  while(acc>=1){
    update();
    updateParticles();
    updateTrail();
    acc-=1;
  }
  draw(); requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
"""

final_html = template.replace('__WEIGHTS_JSON__', weights_json)

for path in ['/home/fahim/game/Air Hockey.html', '/home/fahim/game/air-hockey.html', '/home/fahim/game/www/index.html']:
    with open(path, 'w') as f:
        f.write(final_html)

print('Successfully generated and written perfected HTML files!')

/* Atari57 Research Console — frontend application logic.
 *
 * Talks to the FastAPI sidecar at the same origin (`/api/*`).
 * Falls back to fixtures so the page is still readable when opened via file://.
 *
 * Phase status:
 *   ✓ Phase 1: real game grid, real algorithm rail, selection state, top-bar labels
 *   ☐ Phase 2: WebSocket eval streaming (replaces the local pong-ish sim + action bars)
 *   ☐ Phase 3: tensorboard tailing for the episode-return chart and event log
 *   ☐ Phase 4: replay scrubber wired to served MP4s
 */

(() => {
  "use strict";

  /* ───────── module state ───────── */

  const state = {
    algorithms: [], // [{module, label, family, family_label, checkpoint_count, games_with_checkpoint}]
    games: [],      // [{name, code, has_checkpoint}]
    checkpoints: [], // [{filename, algo_module, algo_label, game, iter, ...}]
    selected: {
      algo_module: "rainbow", // start matching the prototype's default
      game: "Pong",
    },
  };

  /* ───────── api ───────── */

  async function fetchJSON(path) {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`${path} → HTTP ${resp.status}`);
    return resp.json();
  }

  async function loadCatalog() {
    try {
      const [games, algos, ckpts] = await Promise.all([
        fetchJSON("/api/games"),
        fetchJSON("/api/algorithms"),
        fetchJSON("/api/checkpoints"),
      ]);
      state.games = games.items;
      state.algorithms = algos.items;
      state.checkpoints = ckpts.items;
      return true;
    } catch (err) {
      console.warn("API not reachable, falling back to fixtures.", err);
      state.games = FALLBACK_GAMES;
      state.algorithms = FALLBACK_ALGOS;
      state.checkpoints = [];
      return false;
    }
  }

  /* ───────── algorithm rail ───────── */

  const FAMILY_ORDER = ["value", "distributional", "policy"];

  function renderAlgoRail() {
    const root = document.getElementById("algo-rail");
    if (!root) return;
    root.innerHTML = "";

    const grouped = new Map();
    for (const algo of state.algorithms) {
      if (!grouped.has(algo.family)) grouped.set(algo.family, []);
      grouped.get(algo.family).push(algo);
    }

    for (const family of FAMILY_ORDER) {
      const items = grouped.get(family);
      if (!items?.length) continue;

      const family_label = items[0].family_label;
      const groupEl = document.createElement("div");
      groupEl.className = "algo-group";
      groupEl.innerHTML = `<div class="gh">${family_label}</div>`;

      for (const algo of items) {
        const item = document.createElement("div");
        item.className = "algo";
        item.dataset.module = algo.module;
        if (algo.module === state.selected.algo_module) item.classList.add("on");

        const glyph = algo.module === state.selected.algo_module ? "▶" : "▸";
        const perf = algo.checkpoint_count > 0
          ? `<span class="perf">✓ ${algo.checkpoint_count}</span>`
          : `<span class="perf dim">—</span>`;
        item.innerHTML = `
          <span class="glyph">${glyph}</span>
          <span class="name">${algo.label}</span>
          ${perf}
        `;
        item.title = algo.checkpoint_count > 0
          ? `Trained checkpoints: ${algo.games_with_checkpoint.join(", ")}`
          : `No bundled checkpoint for ${algo.label}`;
        item.addEventListener("click", () => selectAlgo(algo.module));
        groupEl.appendChild(item);
      }

      root.appendChild(groupEl);
    }

    // Update count chip in the panel head.
    const chip = document.querySelector('.panel-head .chip');
    if (chip) chip.textContent = String(state.algorithms.length);
  }

  function selectAlgo(module) {
    state.selected.algo_module = module;
    document.querySelectorAll("#algo-rail .algo").forEach((el) => {
      const isOn = el.dataset.module === module;
      el.classList.toggle("on", isOn);
      const glyph = el.querySelector(".glyph");
      if (glyph) glyph.textContent = isOn ? "▶" : "▸";
    });
    const algo = state.algorithms.find((a) => a.module === module);
    if (algo) {
      const label = document.querySelector('header.bar .cell .val[data-field="algo"]');
      if (label) label.textContent = algo.label;
      // Top-bar experiment id like "e_0428_rainbow_pong" — synth from selection
      updateExperimentLabel();
    }
  }

  /* ───────── game grid ───────── */

  function renderGameGrid() {
    const grid = document.getElementById("game-grid");
    if (!grid) return;
    grid.innerHTML = "";

    for (const game of state.games) {
      const cell = document.createElement("div");
      cell.className = "game-cell";
      if (game.has_checkpoint) cell.classList.add("trained");
      if (game.name === "MontezumaRevenge" && !game.has_checkpoint) {
        // backward compat with prototype: keep the "expert" class for hard-exploration games
        cell.classList.add("expert");
      }
      if (game.name === state.selected.game) cell.classList.add("on");
      cell.dataset.game = game.name;
      cell.textContent = game.code;
      cell.title = game.has_checkpoint
        ? `${game.name} — bundled checkpoint available`
        : `${game.name} — no bundled checkpoint`;
      cell.addEventListener("click", () => selectGame(game.name));
      grid.appendChild(cell);
    }

    // Update the "X / 57" count chip showing how many games we have any checkpoint for.
    const chips = document.querySelectorAll(".panel-head .chip");
    const trained = state.games.filter((g) => g.has_checkpoint).length;
    chips.forEach((chip) => {
      if (chip.textContent.includes("/ 57")) {
        chip.textContent = `${trained} / 57`;
      }
    });
  }

  function selectGame(name) {
    state.selected.game = name;
    document.querySelectorAll("#game-grid .game-cell").forEach((el) => {
      el.classList.toggle("on", el.dataset.game === name);
    });
    const label = document.getElementById("game-label");
    if (label) label.textContent = name;
    updateExperimentLabel();
  }

  /* ───────── top-bar labels ───────── */

  function updateExperimentLabel() {
    // Synthesize "e_<date>_<algo>_<game>" — a stable id derived from the selection.
    const expEl = document.querySelector('header.bar .val[data-field="experiment"]');
    if (!expEl) return;
    const algo = state.algorithms.find((a) => a.module === state.selected.algo_module);
    const algoStr = algo ? algo.module.replace(/_/g, "") : "?";
    const gameStr = state.selected.game.toLowerCase();
    const datestamp = new Date().toISOString().slice(2, 10).replace(/-/g, "").slice(2);
    expEl.textContent = `e_${datestamp}_${algoStr}_${gameStr}`;
  }

  /* ───────── action distribution (mock — Phase 2 replaces this) ───────── */

  function renderActions() {
    const ACTIONS = [
      ["NOOP", 0.082, -0.18],
      ["FIRE", 0.118, +0.04],
      ["UP", 0.342, +0.61],
      ["RIGHT", 0.061, -0.22],
      ["LEFT", 0.044, -0.31],
      ["DOWN", 0.298, +0.48],
      ["UP-FIRE", 0.027, -0.04],
      ["DOWN-FIRE", 0.028, -0.06],
    ];
    const root = document.getElementById("actions");
    if (!root) return;
    root.innerHTML = "";
    const maxP = Math.max(...ACTIONS.map((a) => a[1]));
    for (const [nm, p, q] of ACTIONS) {
      const w = (p / maxP) * 100;
      const hot = p === maxP;
      root.insertAdjacentHTML(
        "beforeend",
        `<div class="act">
           <span class="nm ${hot ? "hot" : ""}">${nm}</span>
           <div class="bar"><div class="f" style="width:${w.toFixed(1)}%"></div></div>
           <span class="v">${(p * 100).toFixed(1)}%</span>
         </div>
         <div class="act" style="margin-top: -8px; margin-bottom: 2px;">
           <span class="nm dim" style="font-size: 9.5px;">  Q(s,a)</span>
           <div class="bar q"><div class="f" style="width:${(50 + q * 40).toFixed(1)}%"></div></div>
           <span class="v dim">${q >= 0 ? "+" : ""}${q.toFixed(2)}</span>
         </div>`
      );
    }
  }

  /* ───────── episode-return chart (mock — Phase 3 replaces) ───────── */

  function renderChart() {
    const svg = document.getElementById("chart-svg");
    if (!svg) return;
    const W = 1000, H = 400;

    function curve(n, start, end, noise, seed) {
      const pts = [];
      let s = seed;
      const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
      for (let i = 0; i < n; i++) {
        const t = i / (n - 1);
        const ease = 1 - Math.pow(1 - t, 1.8);
        const v = start + (end - start) * ease + (rnd() - 0.5) * noise;
        pts.push([(i / (n - 1)) * W, v]);
      }
      return pts;
    }
    const yMap = (v) => H - ((v + 21) / 42) * H;
    const pathFrom = (pts) => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + yMap(p[1]).toFixed(1)).join(" ");
    const areaFrom = (pts) =>
      "M" + pts[0][0].toFixed(1) + "," + H + " " +
      pts.map((p) => "L" + p[0].toFixed(1) + "," + yMap(p[1]).toFixed(1)).join(" ") +
      " L" + pts[pts.length - 1][0].toFixed(1) + "," + H + " Z";

    const rainbow = curve(180, -19, 12, 4.5, 7);
    const perDqn = curve(180, -20, 16, 4.0, 31);
    const iqn = curve(180, -20, -1, 5.5, 91);

    svg.innerHTML = `
      <defs>
        <linearGradient id="rgrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#86efac" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="#86efac" stop-opacity="0"/>
        </linearGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="2"/></filter>
      </defs>
      <line x1="0" y1="${yMap(0)}" x2="${W}" y2="${yMap(0)}" stroke="#1e5743" stroke-dasharray="2,4"/>
      <line x1="0" y1="${yMap(9.3)}" x2="${W}" y2="${yMap(9.3)}" stroke="#2e7e5e" stroke-dasharray="6,4" opacity="0.7"/>
      <path d="${pathFrom(perDqn)}" fill="none" stroke="#f5b942" stroke-width="1.2" opacity="0.85"/>
      <path d="${pathFrom(iqn)}" fill="none" stroke="#67e8f9" stroke-width="1.2" opacity="0.55"/>
      <path d="${areaFrom(rainbow)}" fill="url(#rgrad)"/>
      <path d="${pathFrom(rainbow)}" fill="none" stroke="#86efac" stroke-width="2" filter="url(#glow)" opacity="0.7"/>
      <path d="${pathFrom(rainbow)}" fill="none" stroke="#b9fbc0" stroke-width="1.4"/>
      <circle cx="${rainbow[rainbow.length - 1][0]}" cy="${yMap(rainbow[rainbow.length - 1][1])}" r="3" fill="#b9fbc0"/>
      <circle cx="${rainbow[rainbow.length - 1][0]}" cy="${yMap(rainbow[rainbow.length - 1][1])}" r="6" fill="none" stroke="#b9fbc0" opacity="0.4"/>
    `;
  }

  /* ───────── event log (mock — Phase 3 replaces) ───────── */

  function renderEventLog() {
    const LOG = [
      ["04:21:03", "evt", "actor.03", "episode_end · ep <b>4128</b> · return <b>+12.0</b> · len 1843"],
      ["04:21:03", "ok", "learner", "checkpoint saved → <b>Rainbow_Pong_2.ckpt</b> <span class=\"dim\">(18.4 MB)</span>"],
      ["04:20:58", "info", "trainer", "td_loss=0.018  q_value=+0.42  grad_norm=0.31"],
      ["04:20:54", "evt", "actor.05", "episode_end · ep 4127 · return +6.0 · len 2114"],
      ["04:20:51", "info", "trainer", "target network synced @ step 1,824,000"],
      ["04:20:42", "evt", "actor.01", "episode_end · ep 4126 · return +14.0 · len 1622"],
      ["04:20:38", "info", "replay", "buffer 874,332 / 1,000,000 <span class=\"dim\">(87.4%)</span>"],
      ["04:20:30", "warn", "actor.07", "stalled · 0.0 reward in 1200 steps · <span class=\"dim\">resetting</span>"],
      ["04:20:22", "evt", "actor.02", "episode_end · ep 4125 · return +9.0 · len 1944"],
      ["04:20:14", "info", "noisy", "σ_w mean=0.21  σ_b mean=0.18"],
      ["04:20:09", "evt", "actor.06", "episode_end · ep 4124 · return +11.0 · len 1701"],
      ["04:19:55", "ok", "eval", "scheduled @ step 2,000,000 · 10k eval steps"],
      ["04:19:48", "info", "trainer", "td_loss=0.021  q_value=+0.39  grad_norm=0.40"],
      ["04:19:34", "err", "actor.04", "ALE warn · render_mode quirk · <span class=\"dim\">non-fatal</span>"],
      ["04:19:28", "evt", "actor.00", "episode_end · ep 4123 · return +8.0 · len 1814"],
    ];
    const logEl = document.getElementById("log");
    if (!logEl) return;
    logEl.innerHTML = "";
    for (const [ts, lvl, src, msg] of LOG) {
      logEl.insertAdjacentHTML(
        "beforeend",
        `<div class="row"><span class="ts">${ts}</span><span class="lvl ${lvl}">${lvl.toUpperCase()}</span><span class="msg"><span class="dim">${src}</span> ${msg}</span></div>`
      );
    }
  }

  /* ───────── live eval streaming via WebSocket ───────── */

  const evalState = {
    ws: null,
    running: false,
    sessionStep: 0,
    sessionReturn: 0.0,
    episodes: 0,
    actionCounts: [],   // per-action counts for the current session
    actionDim: 0,
    actionNames: [],
    pendingFrame: null, // most recent frame (rendered each rAF tick)
    lastEpisodeReturn: null,
  };

  function findCheckpointFor(algoModule, gameName) {
    return state.checkpoints.find((c) => c.algo_module === algoModule && c.game === gameName);
  }

  function updateHud() {
    const $ = (id) => document.getElementById(id);
    if ($("hud-frame")) $("hud-frame").textContent = evalState.sessionStep.toLocaleString();
    if ($("step-count")) $("step-count").textContent = evalState.sessionStep.toLocaleString();
    if ($("rs-step")) $("rs-step").textContent = (evalState.sessionStep / 1000).toFixed(2) + "k";
    if ($("hud-score")) $("hud-score").textContent = (evalState.lastEpisodeReturn ?? evalState.sessionReturn).toFixed(0).padStart(2, "0");
    if ($("hud-score-2")) $("hud-score-2").textContent = String(evalState.episodes).padStart(2, "0");
  }

  function setPlayButtonState(playing) {
    const btn = document.getElementById("play-btn");
    const icon = document.getElementById("play-icon");
    if (!btn || !icon) return;
    btn.classList.toggle("on", playing);
    icon.innerHTML = playing
      ? '<rect x="3" y="2" width="3" height="12" fill="currentColor"/><rect x="10" y="2" width="3" height="12" fill="currentColor"/>'
      : '<path d="M3 2l11 6-11 6z" fill="currentColor"/>';
  }

  function paintPlaceholder(ctx, W, H, message) {
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#163a2e";
    for (let y = 4; y < H; y += 6) ctx.fillRect(W / 2 - 1, y, 2, 3);
    if (message) {
      ctx.fillStyle = "#4ade80";
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.textAlign = "center";
      ctx.fillText(message, W / 2, H / 2);
    }
  }

  function startGameSim() {
    const c = document.getElementById("game-canvas");
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    const W = c.width, H = c.height;

    paintPlaceholder(ctx, W, H, "▸ press play to run a real eval");

    // Render most recent streamed frame on each animation tick.
    function tick() {
      if (evalState.pendingFrame) {
        const img = evalState.pendingFrame;
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, W, H);
        // Atari frames are 160x210; preserve aspect by letterboxing.
        const scale = Math.min(W / img.width, H / img.height);
        const dw = img.width * scale, dh = img.height * scale;
        const dx = (W - dw) / 2, dy = (H - dh) / 2;
        ctx.drawImage(img, dx, dy, dw, dh);
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    const btn = document.getElementById("play-btn");
    if (btn) {
      btn.addEventListener("click", () => {
        if (evalState.running) {
          stopEvalStream();
        } else {
          startEvalStream(ctx, W, H);
        }
      });
    }
  }

  function startEvalStream(ctx, W, H) {
    const ckpt = findCheckpointFor(state.selected.algo_module, state.selected.game);
    if (!ckpt) {
      paintPlaceholder(ctx, W, H, `no checkpoint for ${state.selected.algo_module}/${state.selected.game}`);
      return;
    }

    paintPlaceholder(ctx, W, H, `connecting · ${ckpt.filename}`);
    const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${wsProto}//${location.host}/api/eval/stream`);
    evalState.ws = ws;
    evalState.running = true;
    evalState.sessionStep = 0;
    evalState.sessionReturn = 0.0;
    evalState.episodes = 0;
    evalState.lastEpisodeReturn = null;
    evalState.actionCounts = [];
    evalState.actionDim = 0;
    evalState.actionNames = [];
    setPlayButtonState(true);

    ws.addEventListener("open", () => {
      ws.send(JSON.stringify({
        action: "start",
        checkpoint: ckpt.filename,
        num_steps: 5000,
        frame_stride: 2,
      }));
    });

    ws.addEventListener("message", (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }

      if (msg.type === "init") {
        evalState.actionDim = msg.action_dim;
        evalState.actionNames = msg.actions;
        evalState.actionCounts = new Array(msg.action_dim).fill(0);
        renderLiveActions();
      } else if (msg.type === "step") {
        evalState.sessionStep = msg.step;
        evalState.sessionReturn = msg.ep_return;
        if (msg.action != null && evalState.actionCounts[msg.action] != null) {
          evalState.actionCounts[msg.action] += 1;
        }
        if (msg.frame_b64) {
          const img = new Image();
          img.onload = () => { evalState.pendingFrame = img; };
          img.src = "data:image/png;base64," + msg.frame_b64;
        }
        if (msg.step % 30 === 0) renderLiveActions();
        updateHud();
      } else if (msg.type === "episode") {
        evalState.episodes += 1;
        evalState.lastEpisodeReturn = msg.episode_return;
        appendLog("evt", "eval", `episode_end · return ${msg.episode_return.toFixed(1)} · steps ${msg.episode_steps}`);
        updateHud();
      } else if (msg.type === "done") {
        appendLog("ok", "eval", `done · ${msg.total_steps} env steps`);
        stopEvalStream();
      } else if (msg.type === "error") {
        appendLog("err", "eval", msg.message);
        paintPlaceholder(ctx, W, H, `error: ${msg.message}`);
        stopEvalStream();
      } else if (msg.type === "started") {
        appendLog("ok", "eval", `subprocess pid=${msg.pid} · ${msg.algo} · ${msg.game}`);
      }
    });

    ws.addEventListener("close", () => {
      evalState.ws = null;
      evalState.running = false;
      setPlayButtonState(false);
    });

    ws.addEventListener("error", () => {
      paintPlaceholder(ctx, W, H, "websocket error — see console");
    });
  }

  function stopEvalStream() {
    const ws = evalState.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ action: "stop" })); } catch {}
      ws.close();
    }
    evalState.running = false;
    evalState.ws = null;
    setPlayButtonState(false);
  }

  function renderLiveActions() {
    if (!evalState.actionDim) return;
    const root = document.getElementById("actions");
    if (!root) return;
    const total = evalState.actionCounts.reduce((a, b) => a + b, 0) || 1;
    const maxP = Math.max(...evalState.actionCounts) / total || 1;
    root.innerHTML = "";
    for (let i = 0; i < evalState.actionDim; i++) {
      const p = evalState.actionCounts[i] / total;
      const w = (p / maxP) * 100;
      const hot = i === evalState.actionCounts.indexOf(Math.max(...evalState.actionCounts));
      root.insertAdjacentHTML(
        "beforeend",
        `<div class="act">
           <span class="nm ${hot ? "hot" : ""}">${evalState.actionNames[i] || ("A" + i)}</span>
           <div class="bar"><div class="f" style="width:${w.toFixed(1)}%"></div></div>
           <span class="v">${(p * 100).toFixed(1)}%</span>
         </div>`
      );
    }
  }

  function appendLog(level, source, message) {
    const logEl = document.getElementById("log");
    if (!logEl) return;
    const ts = new Date().toTimeString().slice(0, 8);
    logEl.insertAdjacentHTML(
      "afterbegin",
      `<div class="row"><span class="ts">${ts}</span><span class="lvl ${level}">${level.toUpperCase()}</span><span class="msg"><span class="dim">${source}</span> ${message}</span></div>`
    );
    // Cap log length so it doesn't grow unbounded.
    while (logEl.children.length > 60) logEl.removeChild(logEl.lastChild);
  }

  /* ───────── header FPS / uptime ───────── */

  function startClock() {
    let t0 = performance.now();
    let frames = 0, lastSec = t0, fps = 60;
    let uptime = 4 * 3600 + 21 * 60 + 8;
    function tick(t) {
      frames++;
      if (t - lastSec >= 1000) {
        fps = (frames * 1000) / (t - lastSec);
        frames = 0; lastSec = t; uptime++;
        const h = String(Math.floor(uptime / 3600)).padStart(2, "0");
        const m = String(Math.floor((uptime % 3600) / 60)).padStart(2, "0");
        const s = String(uptime % 60).padStart(2, "0");
        const u = document.getElementById("uptime");
        const f = document.getElementById("fps");
        if (u) u.textContent = `${h}:${m}:${s}`;
        if (f) f.textContent = fps.toFixed(1);
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ───────── chip / tab toggles (visual feedback) ───────── */

  function wireToggles() {
    const groups = [
      ".chart-tab",
      ".act-toggle .opt",
      ".transport .speed .opt",
    ];
    for (const sel of groups) {
      document.querySelectorAll(sel).forEach((el) => {
        el.addEventListener("click", () => {
          document.querySelectorAll(sel).forEach((x) => x.classList.remove("on"));
          el.classList.add("on");
        });
      });
    }
  }

  /* ───────── fixtures (used only when API is unreachable) ───────── */

  const FALLBACK_GAMES = [
    "Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis", "BankHeist",
    "BattleZone", "BeamRider", "Berzerk", "Bowling", "Boxing", "Breakout", "Centipede",
    "ChopperCommand", "CrazyClimber", "Defender", "DemonAttack", "DoubleDunk", "Enduro",
    "FishingDerby", "Freeway", "Frostbite", "Gopher", "Gravitar", "Hero", "IceHockey",
    "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", "MontezumaRevenge", "MsPacman",
    "NameThisGame", "Phoenix", "Pitfall", "Pong", "PrivateEye", "Qbert", "Riverraid",
    "RoadRunner", "Robotank", "Seaquest", "Skiing", "Solaris", "SpaceInvaders",
    "StarGunner", "Surround", "Tennis", "TimePilot", "Tutankham", "UpNDown", "Venture",
    "VideoPinball", "WizardOfWor", "YarsRevenge", "Zaxxon",
  ].map((name) => ({ name, code: name.slice(0, 3).toUpperCase(), has_checkpoint: name === "Pong" || name === "MontezumaRevenge" }));

  const FALLBACK_ALGOS = [
    ["dqn", "DQN", "value"], ["double_dqn", "Double DQN", "value"],
    ["prioritized_dqn", "Prioritized DQN", "value"], ["drqn", "DRQN", "value"],
    ["r2d2", "R2D2", "value"], ["ngu", "NGU", "value"], ["agent57", "Agent57", "value"],
    ["c51_dqn", "C51 DQN", "distributional"], ["rainbow", "Rainbow", "distributional"],
    ["qr_dqn", "QR-DQN", "distributional"], ["iqn", "IQN", "distributional"],
    ["reinforce", "REINFORCE", "policy"], ["reinforce_baseline", "REINFORCE+B", "policy"],
    ["actor_critic", "Actor-Critic", "policy"], ["a2c", "A2C", "policy"],
    ["sac", "SAC", "policy"], ["ppo", "PPO", "policy"], ["ppo_icm", "PPO+ICM", "policy"],
    ["ppo_rnd", "PPO+RND", "policy"], ["impala", "IMPALA", "policy"],
  ].map(([module, label, family]) => ({
    module, label, family,
    family_label: { value: "VALUE-BASED", distributional: "DISTRIBUTIONAL", policy: "POLICY-BASED" }[family],
    checkpoint_count: 0,
    games_with_checkpoint: [],
  }));

  /* ───────── init ───────── */

  async function init() {
    await loadCatalog();
    renderAlgoRail();
    renderGameGrid();
    renderActions();
    renderChart();
    renderEventLog();
    startGameSim();
    startClock();
    wireToggles();
    selectAlgo(state.selected.algo_module);
    selectGame(state.selected.game);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

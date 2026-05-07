# Atari57 Research Console — Frontend Prototype

A hi-fi single-screen design prototype for the Atari57 research sandbox.

![status: prototype](https://img.shields.io/badge/status-prototype-amber)
![data: mock](https://img.shields.io/badge/data-mock-orange)

## What this is

A CRT/phosphor-aesthetic research console for visualizing deep RL training on Atari games. Single-page app, vanilla HTML/CSS/JS, no build step, no backend.

Designed in [Claude Design](https://claude.ai/design) and implemented as a static prototype.

## What it shows

- **Top status bar** — experiment id, algorithm, game, seed, device, FPS, uptime, save state, tensorboard port
- **Left rail** — algorithm picker (all 19 from `deep_rl_zoo`, grouped by family) + 57-game code grid with state badges (trained / expert / off)
- **Main canvas** — game viewport with mock live frames, transport controls (play / pause / step / speed), replay scrubber, episode-return chart with three agent curves vs. human reference, action distribution + Q-values panel
- **Right inspector** — live hyperparameter readout (learning rate, gamma, epsilon, etc.), Rainbow dueling architecture diagram, agent comparison table, tailing event log
- **Bottom ticker** — running env steps, eval returns, GPU utilization

Layout is fixed-width 1600px (the prototype was specced as a single hi-fi screen, not a responsive design).

## Run it

The page is a single self-contained HTML file. Any static server works:

```bash
# From this directory
python -m http.server 8080
# then open http://localhost:8080
```

Or just open `index.html` directly in a browser (`file://`) — the only network calls are to Google Fonts.

## Data

**Everything is mocked.** The episode-return curves, action distributions, hyperparameters, event log, FPS counter, and uptime are all generated client-side from static fixtures + small JS animations. Wiring this to real training data (reading `runs/` tensorboard event files, polling a Python sidecar, etc.) is out of scope for the prototype — see _Future work_ below.

## Visuals

The aesthetic is intentional retro CRT:
- Phosphor green palette (`#4ade80` and dimmer variants) with amber / red / cyan / magenta accent leds
- Scanline overlay (`body::before`) and vignette (`body::after`) layered above all content
- VT323 for the largest CRT-style numbers, JetBrains Mono for code/labels, Space Grotesk for prose
- Subtle 4.5s flicker animation on the whole `.app`

## Future work (not in scope here)

- **Live data adapter** — a small Python sidecar that tails `runs/<experiment>/events.out.tfevents.*` and serves the metrics over WebSocket. Would replace the mock fixtures.
- **Real game viewport** — connect to the deep_rl_zoo eval loop and stream frames (canvas + image data, or HLS for the recorded MP4s in `recordings/`).
- **Persistent experiment list** — read `checkpoints/` and `runs/` on load to populate the experiment selector with everything actually on disk.
- **Responsive layout** — the prototype is fixed-width 1600px; a real implementation would need breakpoints for laptop screens.

## License

Same as the rest of the repo: Apache License 2.0. The frontend prototype is original work added in this fork; the upstream `deep_rl_zoo` code under `../deep_rl_zoo/` is © Michael Hu (`michaelnny`).

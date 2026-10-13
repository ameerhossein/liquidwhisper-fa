# Interface Design System: Persian ASR Telemetry Cockpit

## Direction and Feel
- **Intent:** High-precision audio telemetry cockpit for AI researchers and engineers monitoring multi-hour ASR training jobs on remote RunPod GPU instances.
- **Atmosphere:** Deep Obsidian and Carbon substrate (`#090b0e` / `#11141b`), restrained Phosphor Cyan accents (`#22d3ee`), Persian Saffron Amber for canary milestones (`#f59e0b`), and Emerald for benchmark gains (`#10b981`).
- **Tactile Behavior:** Fast, responsive transitions (<200ms ease-out), zero layout shifts via tabular numerals, and tactile button press feedback (`scale(0.97)`).

## Depth Strategy & Spacing
- **Depth Approach:** Subtle layered surfaces with whisper-quiet borders:
  - Canvas: `#090b0e`
  - Base Card: `#11141b`
  - Raised Surface: `#171b24`
  - Inset Control: `#0d0f14`
  - Subtle Border: `rgba(255, 255, 255, 0.07)`
  - Medium Border: `rgba(255, 255, 255, 0.12)`
- **Base Unit:** 4px / 8px grid.
  - Component padding: 12px–16px
  - Card padding: 20px
  - Container gutter: 24px–28px
- **Border Radius:**
  - Control / Tag: 6px (`--radius-sm`)
  - Elevated Box: 10px (`--radius-md`)
  - Master Card: 14px (`--radius-lg`)

## Hierarchy Decisions
- **Focal Element:** The **Epoch Benchmark Matrix Table**, clearly delineating model generations and showing normalized WER improvements against baseline Whisper Large v3 with emerald badges.
- **Secondary Focal:** Real-time Telemetry Stat Horizon (Loss, Learning Rate, VRAM, Step, ETA) paired with dynamic SVG loss trajectories.
- **Signature Component:** The **Canary Audio Orthography Diff**, visually highlighting how the model decodes ZWNJ compounds (`می‌گذارند`) and colloquial contractions (`می‌خوام برم`) side-by-side across epochs.
- **Typography:**
  - Monospace & Tabular numerals: `font-variant-numeric: tabular-nums`
  - Type scale ratio: 1.25 (Caption: 11px, Body: 13px, Subhead: 15px, Hero metric: 20px)

## Key Component Patterns
- **Status Pill:** Rounded 20px pill with pulsating LED dot (`#22d3ee` for training, `#f59e0b` for evaluation, `#10b981` for idle).
- **VRAM Gauge:** Inset horizontal fill bar with explicit GB fraction readout.
- **Delta Badge:** Small 4px-radius pill indicating improvement (`↓ -14.2%` with `#10b981` fill) vs. degradation.
- **Benchmark Row:** Elevated hover feedback (`#1e232f`) with subtle highlighting for the best checkpoint row.

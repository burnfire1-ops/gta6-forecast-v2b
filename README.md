# gta6-forecast-v2b

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/burnfire1-ops/gta6-forecast-v2b/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/burnfire1-ops/gta6-forecast-v2b/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

GTA VI launch-day Twitch viewers forecaster (v2-B). One hour into release day, it forecasts the viewers in the "Grand Theft Auto VI" Twitch category for hours 2-24, five quantiles (5/25/50/75/95%) per hour.

## How it works

Four-component ensemble, combined by the median of components in log space:

1. Kernel-weighted log-ratio comparables (Gaussian kernel on log hour-1, bw 1.0).
2. Per-hour log-log OLS regression, floored by a bin-median peak anchor.
3. Midnight-unlock ramp template (US East step at hour 5, US West step at hour 9).
4. Pooled launch shape (median ratio curve, stabilizer).

v2 adds a scale-band filter: the ensemble median is computed from comparables whose hour-1 is within +/-1.0 log units of the target's hour-1 (widened to +/-1.5 if fewer than 5 in band, else the full pool). Spread comes from localized, centered CQR: leave-one-launch-out residuals of the ensemble median, grouped by hour-group, kernel-weighted by log hour-1, with the residual median centred out so q50 stays the ensemble median.

## Results

- Practice (4 cases): mean pinball loss 0.1948, coverage_90 0.957.
- Rolling origin, GTA VI band (p1 200k-600k, n=5): mean loss 0.1381, median error 21.4%, coverage_90 0.991.
- House baseline: 0.387.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Python 3, standard library only, offline, deterministic, <2 s per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=ej1kkdkc&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

# GTA VI Twitch CCU Forecast — hour-by-hour launch-day audience

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![make.fun · rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ranstar04/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![make.fun · reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ranstar04/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Grand Theft Auto VI releases 19 November 2026. One hour into its release day,
this program forecasts how many people will be watching it live on Twitch in
each of the remaining 23 hours (UTC), as five quantiles an hour (5%, 25%, 50%,
75%, 95%) — scored by pinball loss on the log scale.

## The model

A launch-day curve is an hour-1 print times a shape. The shape is not flat:
viewers dip in the small UTC hours, then climb as the US unlocks and prime time
arrives. How far it climbs depends on how big hour 1 already is — a huge hour 1
means the launch is already at peak attention, so the curve stays flatter
(log-ratio at hour 19 correlates −0.63 with log hour-1 viewers across past
launches; −0.69 at hour 23). So, per forecast hour *h*, the program fits

```
ln(v_h / v_1) = a_h + b_h · (ln v_1 − mean ln v_1) + noise
```

by least squares over the comparable launches that share GTA VI's launch
family — rolling-midnight console unlock, no early access, no second platform
wave — with the slope shrunk 15% toward 0 and the median centred on the
residuals' median. The five quantiles are the fitted centre plus the residuals'
standard deviation times the normal quantiles. Launches outside the family are
excluded because their curves are driven by a different mechanism (simultaneous
global unlocks are front-loaded with no ramp; early-access launches have a
contaminated hour 1; second-wave launches jump when a later platform unlocks).
Small categories (hour-1 under 20,000 viewers) are excluded too: their hourly
sampling is sparse and their ratios are noise.

Everything is computed from the case file itself at run time — no network, no
hard-coded tables, no clock. Deterministic: same case, same bytes.

## Validation (leave-one-out over the comparable launches)

Same pinball loss on the log scale the contest scores:

| population | house baseline | this model |
|---|---|---|
| GTA-VI-like launches (n=14) | 0.269 | **0.177** (−34%) |
| all big launches, hour-1 ≥ 20k (n=27) | 0.321 | **0.252** (−22%) |

The house baseline's published practice loss is 0.387. The model was tuned
only on the comparables (leave-one-out); the four practice launches were never
used for fitting.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `*.in` case in the
input directory and writes one `<id>.out` per case: `{"hours":
[{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`, 23 rows, hours 2–24.
Runs in ~0.2 s a case, well inside the 8 s cap.

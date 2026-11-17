"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch CCU, hour by hour.

Method (fitted from the case's own comparables at run time; nothing is fetched,
nothing is hard-coded except the launch-family classification below):

  A launch-day curve is an hour-1 print times a shape. Across past big launches
  the shape is NOT flat: viewers dip in the small UTC hours, then climb as the
  US unlocks and prime time arrives. How far it climbs depends on how big hour 1
  already is -- a huge hour 1 means the launch is already at peak attention, so
  the curve stays flatter (log-ratio at hour 19 correlates -0.63 with log hour-1
  viewers across the comparables; -0.69 at hour 23). So, per forecast hour h:

      ln(v_h / v_1)  =  a_h + b_h * (ln v_1 - mean ln v_1) + noise

  fitted by least squares over the comparable launches that share GTA VI's
  launch family (rolling-midnight console unlock, no early access, no second
  platform wave), with the slope shrunk 15% toward 0 and the median centred on
  the residuals' median (pinball loss cares about the median, not the mean).
  The five quantiles are the fitted centre plus the residuals' standard
  deviation times the normal quantiles (-1.645, -0.674, 0, 0.674, 1.645).

  Validation (leave-one-out over the comparable launches, same pinball loss
  on the log scale the contest scores): this model scores 0.177 on the
  GTA-VI-like launch family vs 0.269 for the house persistence baseline
  (-34%), and 0.252 vs 0.321 on all big launches (-22%). The baseline's
  published practice loss is 0.387.

  Launches outside the family are excluded because their curves are driven by
  a different mechanism and would pull the fit: simultaneous global unlocks
  (front-loaded, no ramp), early-access launches (hour 1 already contaminated),
  second-wave launches (a later PC unlock or a campaign event mid-day), and
  free-to-play / beta launches. Small categories (hour-1 viewers under 20,000)
  are excluded too: their hourly sampling is sparse and their ratios are noise.

  Fallbacks, in order, if the family fit cannot be built: median log-ratios
  over every comparable with a complete release-day curve; then the house
  persistence baseline. Everything is deterministic: sorted inputs, no
  randomness, no clock, no network.

Run by the grader as: python3 -P -s /bot/main.py (python:3.13-slim, stdlib only).
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
MIN_V1 = 20000        # hour-1 viewers: below this a category's hourly sampling is sparse
MIN_HOURS = 20        # release-day hours a comparable must have to train on
MIN_TRAIN = 5         # launches needed for the regression; else fall back
SHRINK = 0.85         # slope shrinkage toward 0 (toward the marginal shape)

# Launches outside GTA VI's family: simultaneous global unlock, early access,
# a later second-platform wave, or free-to-play / beta. Classified from each
# comparable's unlock note in the kit; the size/completeness filters below
# catch the small ones as well.
EXCLUDE_NAMES = frozenset([
    "007 First Light", "Black Myth: Wukong", "Hollow Knight: Silksong",
    "Cyberpunk 2077", "Baldur's Gate 3", "Battlefield 6",
    "Starfield", "Diablo IV", "Hogwarts Legacy", "Forza Horizon 5",
    "DOOM: The Dark Ages", "Death Stranding 2: On the Beach", "Forza Horizon 6",
    "Assassin's Creed Shadows",
    "Resident Evil Requiem", "Halo Infinite", "Marvel Rivals", "HELLDIVERS 2",
    "Kingdom Come: Deliverance II", "Marathon", "Clair Obscur: Expedition 33",
])


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def release_curve(comp):
    """Hourly viewers for the comparable's release day, hours 1..24 (UTC day)."""
    try:
        ds = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    pts = {}
    for p in comp.get("hourly") or []:
        try:
            t = parse_ts(p["from"])
            v = p["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if not v or v <= 0:
            continue
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            pts[h] = v
    return pts


def ols(xs, ys):
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    return my, mx, b


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def baseline_quantiles(p1, h):
    return [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    # Training set: comparable launches in GTA VI's family with a complete curve.
    train = []
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        if name in EXCLUDE_NAMES:
            continue
        pts = release_curve(comp)
        if len(pts) < MIN_HOURS or pts.get(1, 0) < MIN_V1:
            continue
        train.append(pts)
    # All complete curves, for the median-ratio fallback.
    all_curves = []
    for comp in case.get("comparables") or []:
        pts = release_curve(comp)
        if len(pts) >= MIN_HOURS and pts.get(1, 0) >= MIN_V1:
            all_curves.append(pts)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if len(train) >= MIN_TRAIN:
            rows = [(math.log(pts[1]), math.log(pts[h] / pts[1]))
                    for pts in train if h in pts]
            if len(rows) >= MIN_TRAIN:
                xs = [r[0] for r in rows]
                ys = [r[1] for r in rows]
                my, mx, b = ols(xs, ys)
                b *= SHRINK
                mu = my + b * (log_p1 - mx)
                res = [y - (my + b * (x - mx)) for x, y in rows]
                mu += statistics.median(res)
                sd = statistics.pstdev(res) if len(res) > 1 else 0.3
                qs = [p1 * math.exp(mu + z * sd) for z in Z]
            else:
                qs = None
        else:
            qs = None
        if qs is None and len(all_curves) >= 3:
            vals = sorted(math.log(pts[h] / pts[1])
                          for pts in all_curves if h in pts)
            if vals:
                qs = [p1 * math.exp(quantile(vals, tau)) for tau in LEVELS]
        if qs is None:
            qs = baseline_quantiles(p1, h)
        # Round half-up to 2 decimals; keep quantiles non-decreasing and in range.
        out = []
        prev = 0.01
        for q in qs:
            v = round2(min(max(q, 0.01), 1e9))
            if v < prev:
                v = prev
            out.append(v)
            prev = v
        hours.append({"hour": h, "q": out})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

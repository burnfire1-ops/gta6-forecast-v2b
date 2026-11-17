#!/usr/bin/env python3
"""Forecaster v2 for make.fun gta6-twitch-ccu-forecast-01.

v2 = v1 + ONE architectural change: a SCALE-BAND FILTER for the ensemble
pool (thesis v2, Gemini review suggestion: mega-launch filtering).
Everything else is identical to v1.

The change: when computing the ensemble median (the four components),
restrict the comparable pool to launches whose hour-1 is within +/-1.0
log units of the target's hour-1 (a ~2.7x factor). For a target p1=300k
the band is ~110k-810k; for p1=23k (practice p00) it is ~8.5k-62k.
Rationale: v1's median is dragged down by small/average launches, but
GTA VI is a mega-launch; the most relevant comparables are the other
mega-launches (rolling origin: in the GTA VI band p1 200k-600k the best
launches score 0.09-0.12, while the full-pool median is 0.21). The
kernel in v1 already weights by log hour-1 similarity (bw 1.0); the band
is a harder version of the same idea.

Fallback: if the band has < 5 comparables, widen to +/-1.5 log units;
if still < 5, use the full pool (v1 behaviour). Never crashes, never
invalid output.

CQR variants (tested, see test report): (A) CQR residuals from the
band-filtered pool (fully scale-stratified); (B) CQR residuals from
the FULL pool (calibration stays large and stable). Selected via the
V2_CQR_VARIANT env var for testing; the final program hardcodes the
winner. Default (and the safe choice when in doubt) is B, because CQR
needs >= 20 residuals per hour-group and the band pool is small.

Unchanged from v1: the four components, median-of-components (equal
weights), bin-median peak floor, midnight-unlock ramp, quantile
monotonicity by construction + sort, output format, determinism
(no clock, no randomness, comparables sorted by name), runtime target
< 2 s/case. Standard library only, offline.
"""
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))

# --- embedded pooled median shape (research; fallback if no comparables) ---
FALLBACK_SHAPE = {
    2: 0.871, 3: 0.754, 4: 0.733, 5: 0.747, 6: 0.788, 7: 0.802, 8: 0.899,
    9: 0.882, 10: 0.912, 11: 0.928, 12: 0.978, 13: 1.059, 14: 1.181,
    15: 1.518, 16: 1.652, 17: 1.771, 18: 1.951, 19: 2.402, 20: 2.372,
    21: 2.33, 22: 2.298, 23: 2.182, 24: 2.039,
}
KERNEL_BW = 1.0          # log-units bandwidth for the kernel (research: ~1.0)
# --- v2 scale band (the only architectural change vs v1) ---
BAND_HALF_WIDTH = 1.0    # log-units half-width of the scale band (~2.7x)
BAND_WIDE_WIDTH = 1.5    # widened band if the first band has < BAND_MIN_N
BAND_MIN_N = 5           # minimum comparables in a band; else widen / full pool
# CQR variant: "B" = residuals from the FULL pool (winner of the A/B test:
# B beat A on practice 0.1948 vs 0.2003 and on the GTA VI band 0.1381 vs
# 0.1492; hardcoded after testing, no env var in the final program).
CQR_VARIANT = "B"
STEP_H5 = 1.25           # US East unlock step at hour 5 (comparables with a
                         # clear h5 jump have median h5 ratio 0.94 vs pooled
                         # 0.747, i.e. a factor of ~1.26)
STEP_H9 = 1.10           # US West unlock step at hour 9
SIGMA_FALLBACK = 0.15    # house spread, used only when CQR cannot run
# smooth bin-median peak ratio: (log p1, log peak ratio) anchors, from the
# research bin medians (p1 <10k -> ~225x; 10-100k -> ~2.9x;
# 100-400k -> ~1.45x; >400k -> ~1.0x), interpolated in log-log space.
BIN_ANCHORS = [
    (math.log(1000.0), math.log(225.0)),
    (math.log(30000.0), math.log(2.9)),
    (math.log(200000.0), math.log(1.45)),
    (math.log(600000.0), math.log(1.0)),
]
# peak regression: peak ~= 116000 * p1**0.068
# (log peak ratio vs log p1 corr -0.938, n=37)
REG_PEAK_COEF = 116000.0
REG_PEAK_EXP = 0.068
PEAK_RATIO_MAX = 300.0  # sanity clamp on the peak anchor

GROUPS = {
    "trough": (2, 3, 4),
    "ramp": (5, 6, 7, 8),
    "plateau": (9, 10, 11, 12, 13),
    "rise": (14, 15, 16, 17),
    "peak": (18, 19, 20, 21, 22),
    "late": (23, 24),
}
GROUP_OF = {}
for _g, _hs in GROUPS.items():
    for _h in _hs:
        GROUP_OF[_h] = _g


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return None
    if n % 2 == 1:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def quantile(sorted_vals, q):
    """Linear-interpolation quantile of an already-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def weighted_median(pairs):
    """pairs: list of (value, weight). Returns the weighted median value."""
    pairs = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= 0.5 * total:
            return v
    return pairs[-1][0]


def extract_curves(case):
    """Return a list of comparable launch curves, sorted by name.

    Each curve: {"name": str, "v1": float, "hours": {hour: viewers}} for
    hours 1..24 of the comparable's release day. A comparable is kept
    only if it has a valid positive hour-1 (this also drops the GTA VI
    pre-release comparable, whose day_start is the Aug 27 premiere, not
    a launch day, so it has no hour-1).
    """
    curves = []
    for comp in case.get("comparables", []) or []:
        try:
            day_start = parse_ts(comp["day_start"])
        except Exception:
            continue
        hours = {}
        for row in comp.get("hourly", []) or []:
            try:
                t = parse_ts(row["from"])
                v = row.get("viewers")
            except Exception:
                continue
            if v is None:
                continue
            h = int((t - day_start).total_seconds() / 3600.0) + 1
            if 1 <= h <= 24 and v > 0:
                hours[h] = float(v)
        v1 = hours.get(1)
        if v1 and v1 > 0:
            curves.append({"name": str(comp.get("name", "?")), "v1": v1,
                           "hours": hours})
    curves.sort(key=lambda c: c["name"])
    return curves


def band_filter(curves, p1):
    """v2 scale-band filter: restrict the comparable pool to launches whose
    hour-1 is within BAND_HALF_WIDTH log units of the target's hour-1
    (a ~2.7x factor). If that band has fewer than BAND_MIN_N comparables,
    widen to BAND_WIDE_WIDTH; if still too few, return the full pool
    (v1 behaviour). Never returns fewer than the full pool can give;
    deterministic (input order is name-sorted)."""
    log_p1 = math.log(max(p1, 1.0))
    for width in (BAND_HALF_WIDTH, BAND_WIDE_WIDTH):
        band = [c for c in curves
                if abs(math.log(c["v1"]) - log_p1) <= width]
        if len(band) >= BAND_MIN_N:
            return band
    return list(curves)


def pooled_shape(curves):
    """Unweighted median ratio curve {hour: median(viewers_h / viewers_1)}."""
    shape = {}
    for h in HOURS:
        ratios = [c["hours"][h] / c["v1"] for c in curves if h in c["hours"]]
        m = median(ratios)
        shape[h] = m if m is not None else FALLBACK_SHAPE[h]
    return shape


def kernel_curve(curves, p1):
    """Component 1: kernel-weighted median log-ratio curve (as ratios)."""
    log_p1 = math.log(max(p1, 1.0))
    out = {}
    for h in HOURS:
        pairs = []
        for c in curves:
            vh = c["hours"].get(h)
            if vh is None:
                continue
            w = math.exp(-0.5 * ((math.log(c["v1"]) - log_p1) / KERNEL_BW) ** 2)
            pairs.append((math.log(vh / c["v1"]), w))
        wm = weighted_median(pairs)
        out[h] = math.exp(wm) if wm is not None else FALLBACK_SHAPE[h]
    return out


def ols_curve(curves, p1):
    """Component 2 (raw): per-hour OLS of log viewers on log hour-1.

    Returns {hour: predicted viewers}. Falls back to the pooled shape
    for an hour with too few points or zero variance in log hour-1.
    """
    log_p1 = math.log(max(p1, 1.0))
    pooled = pooled_shape(curves)
    out = {}
    for h in HOURS:
        xs, ys = [], []
        for c in curves:
            vh = c["hours"].get(h)
            if vh is None:
                continue
            xs.append(math.log(c["v1"]))
            ys.append(math.log(vh))
        n = len(xs)
        if n < 3:
            out[h] = max(p1, 1.0) * pooled[h]
            continue
        mx = sum(xs) / n
        my = sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        if var <= 0:
            out[h] = max(p1, 1.0) * pooled[h]
            continue
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = cov / var
        a = my - b * mx
        out[h] = math.exp(a + b * log_p1)
    return out


def bin_peak_ratio(p1):
    """Smooth bin-median peak ratio, log-log interpolation of BIN_ANCHORS."""
    x = math.log(max(p1, 1.0))
    pts = BIN_ANCHORS
    if x <= pts[0][0]:
        return math.exp(pts[0][1])
    if x >= pts[-1][0]:
        return math.exp(pts[-1][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return math.exp(y0 + t * (y1 - y0))
    return math.exp(pts[-1][1])


def reg_peak_ratio(p1):
    """Peak ratio from the hour-1 -> peak regression."""
    p1 = max(p1, 1.0)
    return REG_PEAK_COEF * (p1 ** REG_PEAK_EXP) / p1


def target_peak_ratio(p1):
    """Peak anchor: max(regression peak, smooth bin-median peak).

    The regression underpredicts big launches, so the bin median floors
    it; the regression dominates for small hour-1 where the bin median
    is too coarse. Clamped to a sane range.
    """
    r = max(reg_peak_ratio(p1), bin_peak_ratio(p1))
    return min(max(r, 1.0), PEAK_RATIO_MAX)


def step_shape(pooled):
    """Pooled shape with explicit archetype-B unlock steps.

    US East unlock at hour 5 (05:00Z) x1.25, US West at hour 9 (08:00Z)
    a further x1.10. The pooled median washes these steps out because
    launches unlock at different UTC hours; the template puts them back.
    """
    out = {}
    for h in HOURS:
        f = 1.0
        if h >= 5:
            f *= STEP_H5
        if h >= 9:
            f *= STEP_H9
        out[h] = pooled[h] * f
    return out


def rescale_above_one(shape, target_peak):
    """Rescale a ratio shape so its peak equals target_peak, but only the
    part above 1.0 is stretched: the trough (below hour-1) stays anchored
    to hour-1, which is how real launch curves behave (the trough is
    ~0.7-0.9x whatever the peak does).
    """
    peak = max(shape.values())
    if peak <= 1.0:
        # No above-1.0 part: scale uniformly (only reachable if the shape
        # is flat/declining, which the pooled shape never is).
        return {h: v * target_peak for h, v in shape.items()}
    s = (target_peak - 1.0) / (peak - 1.0)
    if s < 0.0:
        s = 0.0
    out = {}
    for h, v in shape.items():
        out[h] = v if v <= 1.0 else 1.0 + (v - 1.0) * s
    return out


def ensemble_components(curves, p1):
    """The four median components, each {hour: viewers}. Returns a dict
    with keys 'kernel', 'regression', 'ramp', 'pooled'."""
    pooled = pooled_shape(curves)
    r_target = target_peak_ratio(p1)

    # 1. kernel-weighted log-ratio comparables
    kern = kernel_curve(curves, p1)
    comp_kernel = {h: max(p1, 1.0) * kern[h] for h in HOURS}

    # 2. per-hour log-log regression, floored by the bin-anchored shape
    ols = ols_curve(curves, p1)
    floor = rescale_above_one(pooled, r_target)
    comp_reg = {h: max(ols[h], max(p1, 1.0) * floor[h]) for h in HOURS}

    # 3. midnight-unlock ramp template, peak rescaled to the same anchor
    stepped = step_shape(pooled)
    ramp = rescale_above_one(stepped, r_target)
    comp_ramp = {h: max(p1, 1.0) * ramp[h] for h in HOURS}

    # 4. pooled launch shape (stabilizer)
    comp_pooled = {h: max(p1, 1.0) * pooled[h] for h in HOURS}

    return {"kernel": comp_kernel, "regression": comp_reg,
            "ramp": comp_ramp, "pooled": comp_pooled}


def ensemble_median(curves, p1):
    """Median of the four components in log space, per hour.

    Equal-weights default per the forecast combination puzzle (the
    thesis allows equal weights or the median of the components; the
    median is the more robust of the two).
    """
    comps = ensemble_components(curves, p1)
    out = {}
    for h in HOURS:
        vals = [math.log(max(comps[k][h], 1e-9))
                for k in ("kernel", "regression", "ramp", "pooled")]
        out[h] = math.exp(median(vals))
    return out


def _weighted_quantile(pairs, q):
    """Weighted quantile (linear interpolation on the weight axis).

    pairs: list of (value, weight). Returns the q-quantile, or None if
    the total weight is not positive.
    """
    pairs = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if total <= 0 or not pairs:
        return None
    target = q * total
    acc = 0.0
    prev_v = pairs[0][0]
    for val, w in pairs:
        new_acc = acc + w
        if new_acc >= target:
            if new_acc == acc:
                return val
            t = (target - acc) / (new_acc - acc)
            return prev_v + t * (val - prev_v)
        prev_v, acc = val, new_acc
    return pairs[-1][0]


def loo_residuals(curves, min_others=5):
    """Leave-one-launch-out residuals of the ensemble median.

    For each comparable j, forecast it from the other comparables and
    record log(actual) - log(ensemble median) per hour. Returns a list
    of (log_p1_j, hour, residual) tuples.
    min_others: minimum comparables remaining after holding one out
    (v1 used 5; v2 variant A passes 3 because the band pool is small).
    """
    out = []
    if len(curves) < min_others + 1:
        return out
    for j, cj in enumerate(curves):
        others = [c for i, c in enumerate(curves) if i != j]
        if len(others) < min_others:
            continue
        try:
            q50 = ensemble_median(others, cj["v1"])
        except Exception:
            continue
        lp = math.log(cj["v1"])
        for h in HOURS:
            y = cj["hours"].get(h)
            if y is None:
                continue
            out.append((lp, h, math.log(y) - math.log(max(q50[h], 1e-9))))
    return out


def cqr_quantiles(curves, p1, q50, min_others=5):
    """Localized, centered CQR fan around the ensemble median.

    Residual quantiles per hour-group from the leave-one-launch-out
    residuals (calibration pool = the curves passed in), weighted by a
    Gaussian kernel on log hour-1 so the calibration set resembles the
    case (localized conformal: the residual distribution of far-away
    launches does not describe this case). The fan is CENTERED: the
    weighted residual median is subtracted from every offset, so q50
    stays the ensemble median and the residuals inform only the spread
    shape. Rationale (verified on the practice cases): the pooled LOO
    residual median is negative (the ensemble overpredicts comparables
    on average) while the practice cases' signed median residuals are
    mostly positive, i.e. the location bias does not transfer across
    launch types, so shifting the median by it actively hurts; the
    spread shape does transfer. Returns {hour: [q05, q25, q50, q75, q95]}.
    min_others is forwarded to loo_residuals (v2 variant A uses 3).
    """
    res = loo_residuals(curves, min_others=min_others)
    log_p1 = math.log(max(p1, 1.0))
    out = {}
    for h in HOURS:
        g = GROUP_OF[h]
        pairs = []
        for lp, hh, r in res:
            if GROUP_OF[hh] != g:
                continue
            w = math.exp(-0.5 * ((lp - log_p1) / KERNEL_BW) ** 2)
            pairs.append((r, w))
        tot = sum(w for _, w in pairs)
        if tot >= 3.0 and len(pairs) >= 20:
            offs = [_weighted_quantile(pairs, tau) for tau in LEVELS]
        else:
            # Fall back to unweighted group residuals, then all residuals.
            grp = sorted(r for _, hh, r in res if GROUP_OF[hh] == g)
            if len(grp) >= 20:
                offs = [quantile(grp, tau) for tau in LEVELS]
            else:
                all_sorted = sorted(r for _, _, r in res)
                if len(all_sorted) >= 20:
                    offs = [quantile(all_sorted, tau) for tau in LEVELS]
                else:
                    # Too few residuals for CQR: house fan.
                    spread = SIGMA_FALLBACK * math.sqrt(h - 1)
                    offs = [z * spread for z in Z]
        if offs[2] is None:
            spread = SIGMA_FALLBACK * math.sqrt(h - 1)
            offs = [z * spread for z in Z]
        # Center: keep the ensemble median as q50.
        med = offs[2]
        offs = [o - med for o in offs]
        qs = [q50[h] * math.exp(off) for off in offs]
        qs = sorted(qs)
        qs = [min(max(q, 1.0), 1e9) for q in qs]
        out[h] = qs
    return out


def house_fan(p1):
    """House-baseline fan around a flat median (fallback only)."""
    p1 = max(p1, 1.0)
    out = {}
    for h in HOURS:
        spread = SIGMA_FALLBACK * math.sqrt(h - 1)
        qs = [p1 * math.exp(z * spread) for z in Z]
        qs = sorted(min(max(q, 1.0), 1e9) for q in qs)
        out[h] = qs
    return out


def forecast_case(case):
    """Forecast one case. Never raises: falls back step by step."""
    # Hour-1 print (known), with defensive fallbacks.
    p1 = None
    try:
        known = case.get("known") or []
        if known:
            p1 = float(known[0].get("viewers"))
    except Exception:
        p1 = None
    if not p1 or p1 <= 0 or not math.isfinite(p1):
        try:
            hh = case.get("history_hourly") or []
            if hh:
                p1 = float(hh[-1].get("viewers"))
        except Exception:
            p1 = None
    if not p1 or p1 <= 0 or not math.isfinite(p1):
        p1 = 100000.0
    p1 = max(p1, 1.0)

    try:
        curves = extract_curves(case)
    except Exception:
        curves = []
    if len(curves) >= 3:
        try:
            # v2: ensemble median from the scale-band pool, not the full pool.
            band = band_filter(curves, p1)
            q50 = ensemble_median(band, p1)
            if CQR_VARIANT == "A":
                # Variant A: CQR residuals from the band-filtered pool
                # (fully scale-stratified).
                quants = cqr_quantiles(band, p1, q50, min_others=3)
            else:
                # Variant B (default): CQR residuals from the FULL pool,
                # so the calibration sample stays large and stable; the
                # kernel weighting inside cqr_quantiles still localizes
                # it to launches near the target's scale.
                quants = cqr_quantiles(curves, p1, q50)
        except Exception:
            quants = None
        if quants is None:
            try:
                band = band_filter(curves, p1)
                q50 = ensemble_median(band, p1)
                quants = {h: [q50[h] * math.exp(z * SIGMA_FALLBACK
                                                 * math.sqrt(h - 1))
                               for z in Z] for h in HOURS}
            except Exception:
                quants = house_fan(p1)
    else:
        # Too few comparables: embedded pooled shape + house fan.
        q50 = {h: p1 * FALLBACK_SHAPE[h] for h in HOURS}
        quants = {}
        for h in HOURS:
            spread = SIGMA_FALLBACK * math.sqrt(h - 1)
            qs = [q50[h] * math.exp(z * spread) for z in Z]
            quants[h] = sorted(min(max(q, 1.0), 1e9) for q in qs)

    rows = []
    for h in HOURS:
        qs = sorted(quants[h])
        qs = [min(max(round(float(q), 2), 1.0), 1e9) for q in qs]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            with open(path) as f:
                case = json.load(f)
            out = forecast_case(case)
        except Exception:
            # Last resort: flat house-style forecast at a neutral level.
            out = {"hours": [
                {"hour": h, "q": sorted(
                    min(max(100000.0 * math.exp(z * SIGMA_FALLBACK
                                                * math.sqrt(h - 1)), 1.0), 1e9)
                    for z in Z)}
                for h in HOURS]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

"""Bed-occupancy forecast: linear trend + day-of-week effect (pure Python)."""
import math
import random


def demo_history(days=90, seed=42):
    rng = random.Random(seed)
    dow = [8, 6, 3, 0, -2, -10, -5]
    return [round(140 + 0.15 * t + dow[t % 7] + rng.gauss(0, 4), 1) for t in range(days)]


def forecast(history, days=7, capacity=200.0):
    n = len(history)
    tm, ym = (n - 1) / 2, sum(history) / n
    slope = sum((t - tm) * (y - ym) for t, y in enumerate(history)) / sum((t - tm) ** 2 for t in range(n))
    icpt = ym - slope * tm
    resid = [y - (icpt + slope * t) for t, y in enumerate(history)]
    effect = []
    for k in range(7):
        r = [resid[t] for t in range(k, n, 7)]
        effect.append(sum(r) / len(r))
    noise = [resid[t] - effect[t % 7] for t in range(n)]
    sd = math.sqrt(sum(x * x for x in noise) / max(n - 2, 1))
    out = []
    for i in range(1, days + 1):
        t = n - 1 + i
        v = max(0.0, icpt + slope * t + effect[t % 7])
        out.append({"day": i, "value": round(v, 1), "low": round(max(0.0, v - 1.96 * sd), 1),
                    "high": round(v + 1.96 * sd, 1)})
    peak = max(out, key=lambda p: p["value"])
    pct = round(100 * peak["value"] / capacity, 1)
    return {"history": history[-30:], "forecast": out, "capacity": capacity,
            "peak": {"day": peak["day"], "value": peak["value"], "percent": pct},
            "alert": pct >= 90, "trend_per_day": round(slope, 2)}

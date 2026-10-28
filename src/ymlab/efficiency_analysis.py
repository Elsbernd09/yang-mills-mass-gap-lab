"""
Descriptive analysis of the paired overrelaxation campaign records.

Decomposes the per-pair ESS/s ratio R = (tau_M / tau_OR) * (t_M / t_OR) into a
decorrelation gain and a cost factor, and compares two aggregation rules.
Descriptive statistics only; no inference beyond the measured seeds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "results" / "runs"


def campaign_dirs(runs: Path | None = None) -> list[Path]:
    runs = Path(runs) if runs else RUNS
    return sorted(
        p for p in runs.glob("final-overrelaxation-efficiency-campaign-*")
        if (p / "campaign_records.csv").exists()
    )


def load_records(path: Path) -> list[dict]:
    with open(Path(path) / "campaign_records.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {"schedule_label": r["schedule_label"]}
        for k, v in r.items():
            if k != "schedule_label":
                d[k] = float(v)
        out.append(d)
    return out


def paired_decomposition(records: list[dict], observable: str = "plaquette") -> list[dict]:
    """One row per (beta, seed, schedule != M) with gain, cost factor, ratio."""
    base = {(r["beta"], r["seed"]): r for r in records if r["overrelaxation_sweeps"] == 0}
    rows = []
    for r in records:
        if r["overrelaxation_sweeps"] == 0:
            continue
        b = base[(r["beta"], r["seed"])]
        gain = b[f"{observable}_tau_int"] / r[f"{observable}_tau_int"]
        cost = b["runtime_seconds"] / r["runtime_seconds"]
        rows.append({
            "beta": r["beta"], "seed": r["seed"], "schedule": r["schedule_label"],
            "sweeps": int(r["overrelaxation_sweeps"]),
            "gain": gain, "cost_factor": cost, "ratio": gain * cost,
            "penalty": 1.0 / cost,
            "ratio_check": r[f"{observable}_ess_per_second"] / b[f"{observable}_ess_per_second"],
        })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean of paired ratios vs ratio of mean ESS/s, per (beta, schedule)."""
    out = []
    keys = sorted({(r["beta"], r["sweeps"], r["schedule"]) for r in rows})
    for beta, sweeps, label in keys:
        sel = [r for r in rows if (r["beta"], r["sweeps"]) == (beta, sweeps)]
        ratios = np.array([r["ratio"] for r in sel])
        gains = np.array([r["gain"] for r in sel])
        costs = np.array([r["cost_factor"] for r in sel])
        out.append({
            "beta": beta, "sweeps": sweeps, "schedule": label, "n": len(sel),
            "mean_of_ratios": float(ratios.mean()),
            "min": float(ratios.min()), "max": float(ratios.max()),
            "mean_gain": float(gains.mean()), "mean_cost_factor": float(costs.mean()),
            "product_of_means": float(gains.mean() * costs.mean()),
        })
    return out


def ratio_of_means(records: list[dict], observable: str = "plaquette") -> dict:
    """Ratio of seed-averaged ESS/s, schedule vs Metropolis, per (beta, sweeps)."""
    res = {}
    betas = sorted({r["beta"] for r in records})
    sweeps = sorted({int(r["overrelaxation_sweeps"]) for r in records})
    key = f"{observable}_ess_per_second"
    for b in betas:
        m0 = np.mean([r[key] for r in records if r["beta"] == b and r["overrelaxation_sweeps"] == 0])
        for s in sweeps:
            if s == 0:
                continue
            ms = np.mean([r[key] for r in records if r["beta"] == b and int(r["overrelaxation_sweeps"]) == s])
            res[(b, s)] = float(ms / m0)
    return res

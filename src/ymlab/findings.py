"""Collect the numbers behind the findings document and paper from data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import efficiency_analysis as EA

ROOT = Path(__file__).resolve().parents[2]


def reproduction_files() -> list[Path]:
    return sorted((ROOT / "results" / "findings").glob("reproduction_*.json"))


def collect() -> dict:
    dirs = EA.campaign_dirs()
    rec = EA.load_records(dirs[-1])
    rows = EA.paired_decomposition(rec)
    agg = EA.aggregate(rows)
    rom = EA.ratio_of_means(rec)
    for a in agg:
        a["ratio_of_means"] = rom[(a["beta"], a["sweeps"])]
    manifest = json.loads((dirs[-1] / "manifest.json").read_text())
    ratios = np.array([r["ratio"] for r in rows])
    gains = np.array([r["gain"] for r in rows])
    pen = np.array([r["penalty"] for r in rows])
    # determinism between the two published campaigns
    det = None
    if len(dirs) >= 2:
        a, b = EA.load_records(dirs[0]), EA.load_records(dirs[1])
        k = lambda r: (r["beta"], r["seed"], r["overrelaxation_sweeps"])
        bm = {k(r): r for r in b}
        det = {
            "tau_max_abs_diff": max(abs(r["plaquette_tau_int"] - bm[k(r)]["plaquette_tau_int"]) for r in a),
            "ess_max_rel_diff": max(abs(r["plaquette_ess_per_second"] / bm[k(r)]["plaquette_ess_per_second"] - 1) for r in a),
            "runs": [dirs[0].name, dirs[1].name],
        }
    repro = []
    for p in reproduction_files():
        d = json.loads(p.read_text())
        r = d["rows"]
        m = [x for x in r if x["sweeps"] == 0]
        h = [x for x in r if x["sweeps"] > 0]
        repro.append({
            "file": p.name, "local_env": d["local_environment"],
            "published_env": d["published_environment"],
            "metropolis_tau_max_diff": max(x["tau_abs_diff"] for x in m),
            "hybrid_identical": sum(x["tau_abs_diff"] < 1e-9 for x in h),
            "hybrid_total": len(h),
            "hybrid_max_tau_ratio": max(x["tau_local"] / x["tau_published"] for x in h),
            "hybrid_min_tau_ratio": min(x["tau_local"] / x["tau_published"] for x in h),
            "runtime_slowdown_median": float(np.median([x["runtime_local"] / x["runtime_published"] for x in r])),
            "rows": r,
        })
    return {
        "campaign": dirs[-1].name, "manifest": manifest, "n_pairs": len(rows),
        "ratio_max": float(ratios.max()), "ratio_min": float(ratios.min()),
        "n_ratio_above_one": int((ratios > 1).sum()),
        "gain_max": float(gains.max()), "gain_min": float(gains.min()),
        "gains_above_one": int((gains > 1).sum()),
        "penalty_min": float(pen.min()), "penalty_max": float(pen.max()),
        "aggregation": agg, "determinism": det, "reproduction": repro,
        "max_aggregation_gap": max(abs(a["mean_of_ratios"] - a["ratio_of_means"]) / a["ratio_of_means"] for a in agg),
    }

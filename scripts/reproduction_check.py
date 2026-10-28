"""
Re-run a subset of the published campaign on THIS machine and store the result.

Subset: published configuration restricted to seed 2026 (all betas, schedules).
Writes results/findings/reproduction_<platform>.json comparing tau_int (which
the study treats as deterministic) and ESS/s (machine dependent) per condition.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ymlab import efficiency_analysis as EA  # noqa: E402
from ymlab.research_campaign import (  # noqa: E402
    OverrelaxationCampaignConfig, run_overrelaxation_campaign)


def main(seed: int = 2026) -> Path:
    pub_dir = EA.campaign_dirs()[-1]
    manifest = json.loads((pub_dir / "manifest.json").read_text())
    c = manifest["configuration"]
    cfg = OverrelaxationCampaignConfig(
        shape=tuple(c["shape"]), betas=tuple(c["betas"]), seeds=(seed,),
        schedules=tuple(c["schedules"]), epsilon=c["epsilon"],
        thermalization_sweeps=c["thermalization_sweeps"],
        measurement_updates=c["measurement_updates"],
        time_direction=c.get("time_direction", 0))
    records = run_overrelaxation_campaign(cfg, progress=False)
    pub = {(r["beta"], r["seed"], int(r["overrelaxation_sweeps"])): r
           for r in EA.load_records(pub_dir)}
    rows = []
    for r in records:
        p = pub[(r.beta, r.seed, r.overrelaxation_sweeps)]
        rows.append({
            "beta": r.beta, "seed": r.seed, "sweeps": r.overrelaxation_sweeps,
            "tau_local": r.plaquette_tau_int, "tau_published": p["plaquette_tau_int"],
            "tau_abs_diff": abs(r.plaquette_tau_int - p["plaquette_tau_int"]),
            "ess_s_local": r.plaquette_ess_per_second,
            "ess_s_published": p["plaquette_ess_per_second"],
            "runtime_local": r.runtime_seconds, "runtime_published": p["runtime_seconds"],
        })
    out = {
        "published_run": pub_dir.name,
        "published_environment": manifest.get("environment"),
        "local_environment": {
            "platform_system": platform.system(), "platform_machine": platform.machine(),
            "platform_release": platform.release(), "python_version": platform.python_version(),
            "numpy_version": np.__version__},
        "rows": rows,
    }
    dest = ROOT / "results" / "findings"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"reproduction_{platform.system().lower()}_{platform.machine()}.json"
    path.write_text(json.dumps(out, indent=2))
    return path


if __name__ == "__main__":
    print(main())

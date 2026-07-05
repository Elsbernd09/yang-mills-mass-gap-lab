"""
Generate the final manuscript from the newest completed clean-Git campaign.

The manuscript template uses explicit replacement tokens rather than a giant
Python f-string. This avoids nested string-formatting and brace conflicts with
mathematical notation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def latest_clean_campaign(root: Path) -> Path:
    """Return the newest completed campaign executed from a clean Git tree."""
    candidates = sorted(
        [
            path
            for path in root.iterdir()
            if (
                path.is_dir()
                and path.name.startswith(
                    "final-overrelaxation-efficiency-campaign"
                )
            )
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for directory in candidates:
        manifest_path = directory / "manifest.json"

        if not manifest_path.exists():
            continue

        manifest = json.loads(
            manifest_path.read_text()
        )

        git = manifest.get(
            "git",
            {},
        )

        if (
            manifest.get("status") == "completed"
            and git.get("available") is True
            and git.get("dirty") is False
        ):
            return directory

    raise RuntimeError(
        "No completed clean-Git final campaign was found."
    )


def value(row: dict[str, str], key: str) -> float:
    """Read one floating-point CSV field."""
    return float(
        row[key]
    )


def aggregate_table(
    summaries: list[dict[str, str]],
) -> str:
    """Create the manuscript aggregate-results table."""
    lines = [
        "| Beta | Schedule | Plaquette tau_int | Plaquette ESS/s | Plaquette ratio | Scalar tau_int | Scalar ESS/s | Scalar ratio | Mean runtime (s) |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summaries:
        lines.append(
            "| "
            f"{value(row, 'beta'):.3f} | "
            f"{row['schedule_label']} | "
            f"{value(row, 'mean_plaquette_tau_int'):.6f} | "
            f"{value(row, 'mean_plaquette_ess_per_second'):.6f} | "
            f"{value(row, 'mean_plaquette_efficiency_ratio_vs_metropolis'):.6f} | "
            f"{value(row, 'mean_scalar_tau_int'):.6f} | "
            f"{value(row, 'mean_scalar_ess_per_second'):.6f} | "
            f"{value(row, 'mean_scalar_efficiency_ratio_vs_metropolis'):.6f} | "
            f"{value(row, 'mean_runtime_seconds'):.6f} |"
        )

    return "\n".join(
        lines
    )


def winner_summary(
    conclusion: dict,
) -> str:
    """Create the manuscript winner summary."""
    lines = []

    best = conclusion[
        "best_schedule_by_beta"
    ]

    for beta, result in sorted(
        best.items(),
        key=lambda item: float(
            item[0]
        ),
    ):
        lines.append(
            "- "
            f"Beta = {float(beta):.1f}: "
            f"plaquette winner = {result['best_plaquette_schedule']} "
            f"(paired ESS/s ratio "
            f"{result['best_plaquette_efficiency_ratio_vs_metropolis']:.6f}); "
            f"scalar winner = {result['best_scalar_schedule']} "
            f"(paired ESS/s ratio "
            f"{result['best_scalar_efficiency_ratio_vs_metropolis']:.6f})."
        )

    return "\n".join(
        lines
    )


root = Path(
    "results/runs"
)

run_directory = latest_clean_campaign(
    root
)

manifest = json.loads(
    (
        run_directory
        / "manifest.json"
    ).read_text()
)

conclusion = json.loads(
    (
        run_directory
        / "campaign_conclusion.json"
    ).read_text()
)

with (
    run_directory
    / "campaign_records.csv"
).open() as file:
    records = list(
        csv.DictReader(
            file
        )
    )

with (
    run_directory
    / "campaign_summary.csv"
).open() as file:
    summaries = list(
        csv.DictReader(
            file
        )
    )

summaries = sorted(
    summaries,
    key=lambda row: (
        value(
            row,
            "beta",
        ),
        int(
            row[
                "overrelaxation_sweeps"
            ]
        ),
    ),
)

best = conclusion[
    "best_schedule_by_beta"
]

scalar_ratio = float(
    best[
        "2.5"
    ][
        "best_scalar_efficiency_ratio_vs_metropolis"
    ]
)

scalar_improvement = (
    scalar_ratio
    - 1.0
) * 100.0

maximum_microcanonical_error = max(
    float(
        row[
            "maximum_microcanonical_error"
        ]
    )
    for row in records
)

mean_acceptance = sum(
    float(
        row[
            "mean_acceptance_rate"
        ]
    )
    for row in records
) / len(
    records
)

template_path = Path(
    "docs/final_research_paper_template.md"
)

template = template_path.read_text()

replacements = {
    "[[RESULTS_TABLE]]": aggregate_table(
        summaries
    ),
    "[[WINNER_SUMMARY]]": winner_summary(
        conclusion
    ),
    "[[SCALAR_RATIO]]": f"{scalar_ratio:.6f}",
    "[[SCALAR_IMPROVEMENT]]": f"{scalar_improvement:.2f}",
    "[[MICRO_ERROR]]": f"{maximum_microcanonical_error:.12e}",
    "[[MEAN_ACCEPTANCE]]": f"{mean_acceptance:.8f}",
    "[[CONFIG_HASH]]": manifest[
        "configuration_hash"
    ],
    "[[GIT_COMMIT]]": manifest[
        "git"
    ][
        "commit"
    ],
    "[[CAMPAIGN_RUNTIME]]": f"{float(manifest['runtime_seconds']):.6f}",
    "[[PYTHON_VERSION]]": manifest[
        "environment"
    ][
        "python_version"
    ],
    "[[NUMPY_VERSION]]": manifest[
        "environment"
    ][
        "numpy_version"
    ],
}

paper = template

for token, replacement in replacements.items():
    paper = paper.replace(
        token,
        replacement,
    )

unresolved = [
    token
    for token in replacements
    if token in paper
]

if unresolved:
    raise RuntimeError(
        f"Unresolved manuscript tokens: {unresolved}"
    )

output_path = Path(
    "docs/final_research_paper.md"
)

output_path.write_text(
    paper
)

print("=" * 100)
print("FINAL PAPER GENERATED FROM CLEAN RELEASE CAMPAIGN")
print("=" * 100)
print("Campaign:", run_directory)
print("Configuration hash:", manifest["configuration_hash"])
print("Git commit:", manifest["git"]["commit"])
print("Git dirty:", manifest["git"]["dirty"])
print("beta=2.5 scalar winner:", best["2.5"]["best_scalar_schedule"])
print("beta=2.5 scalar ratio:", f"{scalar_ratio:.6f}")
print("beta=2.5 scalar improvement:", f"{scalar_improvement:.2f}%")
print("Paper:", output_path)
print("Paper lines:", len(paper.splitlines()))
print("Paper characters:", len(paper))
print("=" * 100)

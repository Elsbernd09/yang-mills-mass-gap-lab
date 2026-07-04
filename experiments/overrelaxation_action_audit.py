"""
Numerical action-invariance audit for SU(2) microcanonical overrelaxation.

The audit runs nontrivial configurations in 2D, 3D, and 4D.

For every overrelaxation sweep, it records the full Wilson action before and
after the sweep.

A microcanonical sweep should preserve the Wilson action up to floating-point
roundoff.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.overrelaxation import overrelaxation_sweep
from ymlab.wilson_action import wilson_action


def audit_configuration(
    label,
    shape,
    beta,
    epsilon,
    thermalization_sweeps,
    overrelaxation_sweeps,
    seed,
):
    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=seed,
    )

    for _ in range(
        thermalization_sweeps
    ):
        metropolis_sweep(
            lattice=lattice,
            beta=beta,
            epsilon=epsilon,
        )

    rows = []

    for sweep in range(
        overrelaxation_sweeps
    ):
        before = wilson_action(
            lattice,
            beta=beta,
        )

        result = overrelaxation_sweep(
            lattice=lattice,
            beta=beta,
        )

        after = wilson_action(
            lattice,
            beta=beta,
        )

        absolute_error = abs(
            after - before
        )

        relative_error = (
            absolute_error
            / max(
                abs(before),
                np.finfo(float).eps,
            )
        )

        rows.append(
            {
                "label": label,
                "shape": str(shape),
                "dimension": len(shape),
                "sweep": sweep,
                "beta": beta,
                "action_before": before,
                "action_after": after,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "updated_links": (
                    result.updated_links
                ),
                "skipped_links": (
                    result.skipped_links
                ),
                "maximum_local_action_error": (
                    result.maximum_local_action_error
                ),
            }
        )

    return rows


def main() -> None:
    configurations = [
        (
            "2D",
            (6, 6),
            1.7,
            0.18,
            20,
            20,
            2026,
        ),
        (
            "3D",
            (4, 4, 4),
            2.0,
            0.18,
            20,
            20,
            2027,
        ),
        (
            "4D",
            (3, 3, 3, 3),
            2.3,
            0.16,
            15,
            15,
            2028,
        ),
    ]

    print(
        "SU(2) Overrelaxation Action-Invariance Audit"
    )
    print("=" * 106)

    all_rows = []

    for configuration in configurations:
        (
            label,
            shape,
            beta,
            epsilon,
            thermalization_sweeps,
            overrelaxation_sweeps,
            seed,
        ) = configuration

        print()
        print(
            f"Auditing {label} lattice {shape}..."
        )

        rows = audit_configuration(
            label=label,
            shape=shape,
            beta=beta,
            epsilon=epsilon,
            thermalization_sweeps=(
                thermalization_sweeps
            ),
            overrelaxation_sweeps=(
                overrelaxation_sweeps
            ),
            seed=seed,
        )

        all_rows.extend(rows)

        errors = np.asarray(
            [
                row["absolute_error"]
                for row in rows
            ],
            dtype=float,
        )

        relative_errors = np.asarray(
            [
                row["relative_error"]
                for row in rows
            ],
            dtype=float,
        )

        local_errors = np.asarray(
            [
                row[
                    "maximum_local_action_error"
                ]
                for row in rows
            ],
            dtype=float,
        )

        print(
            f"  sweeps audited:             "
            f"{len(rows)}"
        )
        print(
            f"  max global action error:    "
            f"{np.max(errors):.12e}"
        )
        print(
            f"  mean global action error:   "
            f"{np.mean(errors):.12e}"
        )
        print(
            f"  max relative action error:  "
            f"{np.max(relative_errors):.12e}"
        )
        print(
            f"  max local action error:     "
            f"{np.max(local_errors):.12e}"
        )

    maximum_error = max(
        row["absolute_error"]
        for row in all_rows
    )

    passed = bool(
        maximum_error < 1e-8
    )

    print()
    print("=" * 106)

    print(
        f"Maximum global action error:  "
        f"{maximum_error:.12e}"
    )

    print(
        "Overall microcanonical audit: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    data_dir = Path(
        "results/data"
    )
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        data_dir
        / "overrelaxation_action_audit.csv"
    )

    with output_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                all_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    print()
    print(
        f"Saved audit data: {output_path}"
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

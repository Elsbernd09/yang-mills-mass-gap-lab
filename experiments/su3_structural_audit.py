"""
SU(3) finite-lattice structural validation audit.

The audit checks:

1. local-versus-global Wilson-action differences,
2. local gauge invariance of the full Wilson action,
3. average plaquette gauge invariance,
4. closed Wilson-loop gauge invariance,
5. SU(3) link membership after gauge transformation.

The calculations are performed on nontrivial finite SU(3) configurations.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from ymlab.gauge_lattice import GaugeLattice
from ymlab.generic_gauge import (
    generic_average_plaquette,
    generic_metropolis_sweep,
    generic_rectangular_wilson_loop,
    generic_wilson_action,
)
from ymlab.generic_gauge_transformations import (
    gauge_transform_generic_lattice,
    maximum_generic_link_membership_error,
    random_generic_gauge_field,
)
from ymlab.generic_validation import (
    compare_generic_action_differences,
)
from ymlab.group_interface import su3_group


def main() -> None:
    shape = (
        4,
        4,
        4,
    )

    beta = 5.5
    epsilon = 0.05
    thermalization_sweeps = 20
    proposal_checks = 60
    seed = 2026

    print(
        "SU(3) Finite-Lattice Structural Validation Audit"
    )
    print("=" * 112)

    print(
        f"Lattice shape:                 {shape}"
    )

    print(
        f"Beta:                          {beta}"
    )

    print(
        f"Proposal epsilon:              {epsilon}"
    )

    print(
        f"Thermalization sweeps:         "
        f"{thermalization_sweeps}"
    )

    print(
        f"Action-difference checks:      "
        f"{proposal_checks}"
    )

    lattice = GaugeLattice(
        shape=shape,
        group=su3_group(),
        cold_start=True,
        seed=seed,
    )

    acceptance_rates = []

    print()
    print(
        "Generating nontrivial SU(3) configuration..."
    )

    for _ in range(
        thermalization_sweeps
    ):
        result = generic_metropolis_sweep(
            lattice=lattice,
            beta=beta,
            epsilon=epsilon,
        )

        acceptance_rates.append(
            result.acceptance_rate
        )

    rng = np.random.default_rng(
        seed + 1000
    )

    action_rows = []

    for check_index in range(
        proposal_checks
    ):
        site = tuple(
            int(
                rng.integers(
                    0,
                    extent,
                )
            )
            for extent in shape
        )

        mu = int(
            rng.integers(
                0,
                lattice.dim,
            )
        )

        proposal = (
            lattice.group.small_random(
                epsilon,
                rng,
            )
            @ lattice.get_link(
                site,
                mu,
            )
        )

        comparison = (
            compare_generic_action_differences(
                lattice=lattice,
                site=site,
                mu=mu,
                proposal=proposal,
                beta=beta,
                atol=1e-9,
                rtol=1e-9,
            )
        )

        action_rows.append(
            {
                "check_index": check_index,
                "site": str(
                    site
                ),
                "direction": mu,
                "local_difference": (
                    comparison.local_difference
                ),
                "global_difference": (
                    comparison.global_difference
                ),
                "absolute_error": (
                    comparison.absolute_error
                ),
                "relative_error": (
                    comparison.relative_error
                ),
                "consistent": (
                    comparison.consistent
                ),
            }
        )

    action_passed = all(
        row[
            "consistent"
        ]
        for row in action_rows
    )

    maximum_action_error = max(
        row[
            "absolute_error"
        ]
        for row in action_rows
    )

    mean_action_error = float(
        np.mean(
            [
                row[
                    "absolute_error"
                ]
                for row in action_rows
            ]
        )
    )

    print()
    print(
        "Local/global action consistency"
    )
    print("-" * 112)

    print(
        f"Consistent proposals:           "
        f"{sum(row['consistent'] for row in action_rows)}/"
        f"{proposal_checks}"
    )

    print(
        f"Maximum absolute error:         "
        f"{maximum_action_error:.12e}"
    )

    print(
        f"Mean absolute error:            "
        f"{mean_action_error:.12e}"
    )

    print(
        "Action consistency:            "
        f"{'PASS' if action_passed else 'FAIL'}"
    )

    gauge_field = random_generic_gauge_field(
        lattice=lattice,
        rng=np.random.default_rng(
            seed + 2000
        ),
    )

    transformed = (
        gauge_transform_generic_lattice(
            lattice=lattice,
            gauge_field=gauge_field,
        )
    )

    action_before = generic_wilson_action(
        lattice=lattice,
        beta=beta,
    )

    action_after = generic_wilson_action(
        lattice=transformed,
        beta=beta,
    )

    plaquette_before = (
        generic_average_plaquette(
            lattice
        )
    )

    plaquette_after = (
        generic_average_plaquette(
            transformed
        )
    )

    loop_before = (
        generic_rectangular_wilson_loop(
            lattice=lattice,
            site=(0, 1, 2),
            mu=0,
            nu=2,
            width=2,
            height=2,
        )
    )

    loop_after = (
        generic_rectangular_wilson_loop(
            lattice=transformed,
            site=(0, 1, 2),
            mu=0,
            nu=2,
            width=2,
            height=2,
        )
    )

    membership_error = (
        maximum_generic_link_membership_error(
            transformed
        )
    )

    action_gauge_error = abs(
        action_after
        - action_before
    )

    plaquette_gauge_error = abs(
        plaquette_after
        - plaquette_before
    )

    loop_gauge_error = abs(
        loop_after
        - loop_before
    )

    gauge_passed = bool(
        np.isclose(
            action_before,
            action_after,
            atol=1e-8,
            rtol=1e-10,
        )
        and np.isclose(
            plaquette_before,
            plaquette_after,
            atol=1e-10,
            rtol=1e-10,
        )
        and np.isclose(
            loop_before,
            loop_after,
            atol=1e-10,
            rtol=1e-10,
        )
        and membership_error < 1e-8
    )

    print()
    print(
        "SU(3) local gauge-invariance audit"
    )
    print("-" * 112)

    print(
        f"Wilson action before:           "
        f"{action_before:.12e}"
    )

    print(
        f"Wilson action after:            "
        f"{action_after:.12e}"
    )

    print(
        f"Wilson-action gauge error:      "
        f"{action_gauge_error:.12e}"
    )

    print(
        f"Average-plaquette gauge error:  "
        f"{plaquette_gauge_error:.12e}"
    )

    print(
        f"Closed-loop gauge error:        "
        f"{loop_gauge_error:.12e}"
    )

    print(
        f"Maximum link membership error:  "
        f"{membership_error:.12e}"
    )

    print(
        "Gauge-invariance audit:        "
        f"{'PASS' if gauge_passed else 'FAIL'}"
    )

    overall_passed = bool(
        action_passed
        and gauge_passed
    )

    print()
    print("=" * 112)

    print(
        f"Mean thermal acceptance:        "
        f"{np.mean(acceptance_rates):.8f}"
    )

    print(
        "OVERALL SU(3) STRUCTURAL AUDIT: "
        f"{'PASS' if overall_passed else 'FAIL'}"
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
        / "su3_structural_audit.csv"
    )

    with output_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                action_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            action_rows
        )

    print()
    print(
        f"Saved audit data: {output_path}"
    )

    if not overall_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()

"""
Local-versus-global Wilson-action consistency audit.

This experiment tests one of the core numerical identities required by the
local Metropolis update engine:

    Delta S_local = Delta S_global.

For randomly selected links and SU(2) proposals, it compares the staple-based
local action difference with a complete recomputation of the Wilson action.

The audit is run in 2D, 3D, and 4D.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from ymlab.lattice import Lattice
from ymlab.su2 import small_random_su2
from ymlab.validation import compare_action_differences


def audit_dimension(
    label,
    shape,
    beta,
    proposals,
    seed,
):
    lattice = Lattice(
        shape=shape,
        cold_start=False,
        seed=seed,
    )

    rng = np.random.default_rng(
        seed + 10000
    )

    sites = list(
        lattice.sites()
    )

    rows = []

    for proposal_index in range(
        proposals
    ):
        site = sites[
            int(
                rng.integers(
                    0,
                    len(sites),
                )
            )
        ]

        mu = int(
            rng.integers(
                0,
                lattice.dim,
            )
        )

        old_link = lattice.get_link(
            site,
            mu,
        )

        proposal = (
            small_random_su2(
                epsilon=0.18,
                rng=rng,
            )
            @ old_link
        )

        comparison = compare_action_differences(
            lattice=lattice,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=beta,
            atol=1e-10,
            rtol=1e-10,
        )

        rows.append(
            {
                "dimension_label": label,
                "shape": str(shape),
                "proposal_index": proposal_index,
                "site": str(site),
                "direction": mu,
                "beta": beta,
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

    return rows


def main() -> None:
    configurations = [
        (
            "2D",
            (5, 5),
            1.7,
            40,
            2026,
        ),
        (
            "3D",
            (4, 4, 4),
            2.0,
            35,
            2027,
        ),
        (
            "4D",
            (3, 3, 3, 3),
            2.3,
            30,
            2028,
        ),
    ]

    print(
        "Wilson Action Local/Global Consistency Audit"
    )
    print("=" * 100)
    print()

    all_rows = []

    for (
        label,
        shape,
        beta,
        proposals,
        seed,
    ) in configurations:
        print(
            f"Auditing {label} lattice {shape}..."
        )

        rows = audit_dimension(
            label=label,
            shape=shape,
            beta=beta,
            proposals=proposals,
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

        passed = sum(
            bool(row["consistent"])
            for row in rows
        )

        print(
            f"  proposals checked:       {len(rows)}"
        )
        print(
            f"  consistency passes:      "
            f"{passed}/{len(rows)}"
        )
        print(
            f"  maximum absolute error:  "
            f"{np.max(errors):.12e}"
        )
        print(
            f"  mean absolute error:     "
            f"{np.mean(errors):.12e}"
        )
        print(
            f"  maximum relative error:  "
            f"{np.max(relative_errors):.12e}"
        )
        print()

    all_passed = all(
        bool(row["consistent"])
        for row in all_rows
    )

    all_errors = np.asarray(
        [
            row["absolute_error"]
            for row in all_rows
        ],
        dtype=float,
    )

    print("=" * 100)
    print(
        f"Total proposals audited:       "
        f"{len(all_rows)}"
    )
    print(
        f"Maximum absolute delta-S error: "
        f"{np.max(all_errors):.12e}"
    )
    print(
        "Overall action consistency:     "
        f"{'PASS' if all_passed else 'FAIL'}"
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
        / "action_consistency_audit.csv"
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

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

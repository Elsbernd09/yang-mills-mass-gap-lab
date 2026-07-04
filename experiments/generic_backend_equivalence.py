"""
Generic GaugeLattice backend equivalence experiment.

The experiment evolves the established SU(2)-specific lattice implementation,
copies the resulting configuration into the new group-aware GaugeLattice, and
compares the two implementations on the exact same link field.

The comparison includes:

1. plaquette matrices,
2. average plaquette,
3. Wilson action,
4. local staples,
5. local action differences,
6. rectangular Wilson loops.

A separate SU(3) generic-lattice smoke test checks that the new architecture
can carry 3x3 gauge links through cold starts and local Metropolis evolution.

The purpose is regression validation, not physical SU(2)-versus-SU(3)
comparison at matched bare coupling.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from ymlab.gauge_lattice import GaugeLattice
from ymlab.generic_gauge import (
    generic_average_plaquette,
    generic_local_action_difference,
    generic_metropolis_sweep,
    generic_plaquette,
    generic_rectangular_wilson_loop,
    generic_staple,
    generic_wilson_action,
)
from ymlab.group_interface import (
    su2_group,
    su3_group,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.observables import rectangular_wilson_loop
from ymlab.plaquette import (
    average_plaquette,
    plaquette,
)
from ymlab.staples import (
    local_action_difference,
    staple,
)
from ymlab.su2 import small_random_su2
from ymlab.wilson_action import wilson_action


def copy_to_generic(
    lattice,
):
    generic = GaugeLattice(
        shape=lattice.shape,
        group=su2_group(),
        cold_start=True,
        seed=lattice.seed,
    )

    for site in lattice.sites():
        for mu in range(
            lattice.dim
        ):
            generic.set_link(
                site,
                mu,
                lattice.get_link(
                    site,
                    mu,
                ),
            )

    return generic


def main() -> None:
    print(
        "Generic GaugeLattice Backend Equivalence Audit"
    )
    print("=" * 108)

    beta = 2.0
    epsilon = 0.18

    old = Lattice(
        shape=(4, 4, 4),
        cold_start=True,
        seed=2026,
    )

    print()
    print(
        "Generating nontrivial established SU(2) configuration..."
    )

    for _ in range(10):
        metropolis_sweep(
            lattice=old,
            beta=beta,
            epsilon=epsilon,
        )

    generic = copy_to_generic(
        old
    )

    rows = []

    plaquette_cases = [
        ((0, 0, 0), 0, 1),
        ((1, 2, 3), 1, 2),
        ((3, 1, 2), 0, 2),
    ]

    for index, (
        site,
        mu,
        nu,
    ) in enumerate(
        plaquette_cases
    ):
        old_value = plaquette(
            lattice=old,
            site=site,
            mu=mu,
            nu=nu,
        )

        generic_value = (
            generic_plaquette(
                lattice=generic,
                site=site,
                mu=mu,
                nu=nu,
            )
        )

        error = float(
            np.max(
                np.abs(
                    old_value
                    - generic_value
                )
            )
        )

        rows.append(
            {
                "comparison": (
                    f"plaquette_{index}"
                ),
                "old_value": "",
                "generic_value": "",
                "absolute_error": error,
            }
        )

    old_average = average_plaquette(
        old
    )

    generic_average = (
        generic_average_plaquette(
            generic
        )
    )

    rows.append(
        {
            "comparison": "average_plaquette",
            "old_value": old_average,
            "generic_value": generic_average,
            "absolute_error": abs(
                old_average
                - generic_average
            ),
        }
    )

    old_action = wilson_action(
        old,
        beta=beta,
    )

    generic_action = (
        generic_wilson_action(
            generic,
            beta=beta,
        )
    )

    rows.append(
        {
            "comparison": "wilson_action",
            "old_value": old_action,
            "generic_value": generic_action,
            "absolute_error": abs(
                old_action
                - generic_action
            ),
        }
    )

    staple_cases = [
        ((0, 0, 0), 0),
        ((1, 2, 3), 1),
        ((3, 1, 2), 2),
    ]

    for index, (
        site,
        mu,
    ) in enumerate(
        staple_cases
    ):
        old_value = staple(
            lattice=old,
            site=site,
            mu=mu,
        )

        generic_value = (
            generic_staple(
                lattice=generic,
                site=site,
                mu=mu,
            )
        )

        error = float(
            np.max(
                np.abs(
                    old_value
                    - generic_value
                )
            )
        )

        rows.append(
            {
                "comparison": (
                    f"staple_{index}"
                ),
                "old_value": "",
                "generic_value": "",
                "absolute_error": error,
            }
        )

    rng = np.random.default_rng(
        314159
    )

    site = (
        1,
        2,
        3,
    )

    mu = 1

    proposal = (
        small_random_su2(
            epsilon=0.15,
            rng=rng,
        )
        @ old.get_link(
            site,
            mu,
        )
    )

    old_delta = local_action_difference(
        lattice=old,
        site=site,
        mu=mu,
        proposal=proposal,
        beta=beta,
    )

    generic_delta = (
        generic_local_action_difference(
            lattice=generic,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=beta,
        )
    )

    rows.append(
        {
            "comparison": (
                "local_action_difference"
            ),
            "old_value": old_delta,
            "generic_value": generic_delta,
            "absolute_error": abs(
                old_delta
                - generic_delta
            ),
        }
    )

    old_loop = rectangular_wilson_loop(
        lattice=old,
        site=(0, 1, 2),
        mu=0,
        nu=2,
        width=2,
        height=2,
    )

    generic_loop = (
        generic_rectangular_wilson_loop(
            lattice=generic,
            site=(0, 1, 2),
            mu=0,
            nu=2,
            width=2,
            height=2,
        )
    )

    rows.append(
        {
            "comparison": "wilson_loop_2x2",
            "old_value": old_loop,
            "generic_value": generic_loop,
            "absolute_error": abs(
                old_loop
                - generic_loop
            ),
        }
    )

    print()
    print(
        "SU(2) old/new backend equivalence"
    )
    print("-" * 108)

    for row in rows:
        print(
            f"{row['comparison']:<36}"
            f"error = "
            f"{row['absolute_error']:.12e}"
        )

    maximum_error = max(
        float(
            row[
                "absolute_error"
            ]
        )
        for row in rows
    )

    equivalence_passed = bool(
        maximum_error < 1e-9
    )

    print()
    print(
        f"Maximum SU(2) backend error:       "
        f"{maximum_error:.12e}"
    )

    print(
        "SU(2) generic-backend equivalence: "
        f"{'PASS' if equivalence_passed else 'FAIL'}"
    )

    print()
    print(
        "Running generic SU(3) lattice smoke test..."
    )

    su3_lattice = GaugeLattice(
        shape=(3, 3, 3),
        group=su3_group(),
        cold_start=True,
        seed=2027,
    )

    su3_cold_action = (
        generic_wilson_action(
            su3_lattice,
            beta=5.5,
        )
    )

    su3_cold_plaquette = (
        generic_average_plaquette(
            su3_lattice
        )
    )

    acceptance_rates = []

    for _ in range(5):
        result = generic_metropolis_sweep(
            lattice=su3_lattice,
            beta=5.5,
            epsilon=0.05,
        )

        acceptance_rates.append(
            result.acceptance_rate
        )

    su3_final_action = (
        generic_wilson_action(
            su3_lattice,
            beta=5.5,
        )
    )

    su3_final_plaquette = (
        generic_average_plaquette(
            su3_lattice
        )
    )

    su3_membership_passed = all(
        su3_lattice.group.is_member(
            su3_lattice.get_link(
                site,
                mu,
            )
        )
        for site in su3_lattice.sites()
        for mu in range(
            su3_lattice.dim
        )
    )

    print()
    print(
        f"SU(3) cold Wilson action:          "
        f"{su3_cold_action:.12e}"
    )

    print(
        f"SU(3) cold average plaquette:      "
        f"{su3_cold_plaquette:.12f}"
    )

    print(
        f"SU(3) mean Metropolis acceptance:  "
        f"{np.mean(acceptance_rates):.8f}"
    )

    print(
        f"SU(3) final Wilson action:         "
        f"{su3_final_action:.12e}"
    )

    print(
        f"SU(3) final average plaquette:     "
        f"{su3_final_plaquette:.12f}"
    )

    print(
        "SU(3) link membership preserved:   "
        f"{'PASS' if su3_membership_passed else 'FAIL'}"
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
        / "generic_backend_equivalence.csv"
    )

    with output_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        f"Saved equivalence data: {output_path}"
    )

    if (
        not equivalence_passed
        or not su3_membership_passed
    ):
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()

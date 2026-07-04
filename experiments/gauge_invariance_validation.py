"""
Numerical gauge-invariance validation experiment.

This experiment:

1. Generates a nontrivial finite SU(2) lattice configuration.
2. Measures gauge-invariant observables.
3. Generates a random local gauge transformation G(x).
4. Transforms every link by

       U_mu(x) -> G(x) U_mu(x) G(x + mu)^dagger

5. Recomputes the observables.
6. Reports absolute and relative numerical differences.

Gauge invariance is a structural requirement of Yang-Mills theory. This
experiment validates that the finite-lattice implementation respects that
structure up to numerical floating-point tolerance.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from ymlab.gauge_transformations import (
    compare_scalar_observable,
    gauge_transform_lattice,
    max_link_membership_error,
    random_gauge_field,
)
from ymlab.lattice import Lattice
from ymlab.monte_carlo import run_metropolis
from ymlab.observables import (
    average_rectangular_wilson_loop,
    rectangular_wilson_loop,
)
from ymlab.plaquette import (
    average_plaquette,
    average_plaquette_action_density,
)
from ymlab.wilson_action import (
    action_per_plaquette,
    wilson_action,
)


def measure_observables(lattice: Lattice, beta: float) -> dict[str, float]:
    """
    Measure a collection of gauge-invariant scalar observables.
    """
    return {
        "wilson_action": wilson_action(lattice, beta=beta),
        "action_per_plaquette": action_per_plaquette(lattice, beta=beta),
        "average_plaquette": average_plaquette(lattice),
        "average_plaquette_action_density": (
            average_plaquette_action_density(lattice)
        ),
        "wilson_loop_1x1": rectangular_wilson_loop(
            lattice=lattice,
            site=tuple(0 for _ in lattice.shape),
            mu=0,
            nu=1,
            width=1,
            height=1,
        ),
        "wilson_loop_2x2": rectangular_wilson_loop(
            lattice=lattice,
            site=tuple(0 for _ in lattice.shape),
            mu=0,
            nu=1,
            width=2,
            height=2,
        ),
        "average_wilson_loop_1x1": average_rectangular_wilson_loop(
            lattice=lattice,
            mu=0,
            nu=1,
            width=1,
            height=1,
        ),
        "average_wilson_loop_2x2": average_rectangular_wilson_loop(
            lattice=lattice,
            mu=0,
            nu=1,
            width=2,
            height=2,
        ),
    }


def main() -> None:
    shape = (6, 6)
    beta = 2.0
    sweeps = 80
    burn_in = 20
    epsilon = 0.18
    lattice_seed = 2026
    gauge_seed = 314159

    print("Yang-Mills Gauge-Invariance Validation")
    print("=" * 88)
    print(f"Lattice shape:                {shape}")
    print(f"Beta:                         {beta}")
    print(f"Total Metropolis sweeps:      {sweeps}")
    print(f"Burn-in:                      {burn_in}")
    print(f"Proposal epsilon:              {epsilon}")
    print(f"Lattice RNG seed:             {lattice_seed}")
    print(f"Gauge transformation seed:    {gauge_seed}")
    print()

    lattice = Lattice(
        shape=shape,
        cold_start=True,
        seed=lattice_seed,
    )

    print("Generating nontrivial SU(2) lattice configuration...")

    simulation = run_metropolis(
        lattice=lattice,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        measurement_interval=1,
        burn_in=burn_in,
    )

    print(
        "Mean post-burn-in acceptance rate: "
        f"{np.mean(simulation.acceptance_rates):.8f}"
    )
    print()

    before = measure_observables(
        lattice=lattice,
        beta=beta,
    )

    gauge_field = random_gauge_field(
        lattice=lattice,
        seed=gauge_seed,
    )

    transformed = gauge_transform_lattice(
        lattice=lattice,
        gauge_field=gauge_field,
    )

    after = measure_observables(
        lattice=transformed,
        beta=beta,
    )

    comparisons = [
        compare_scalar_observable(
            name=name,
            before=before[name],
            after=after[name],
            atol=1e-8,
            rtol=1e-8,
        )
        for name in before
    ]

    print("Gauge-invariant observable comparison")
    print("-" * 132)
    print(
        f"{'Observable':<38}"
        f"{'Before':>20}"
        f"{'After':>20}"
        f"{'Abs Difference':>20}"
        f"{'Rel Difference':>20}"
        f"{'Status':>12}"
    )
    print("-" * 132)

    for comparison in comparisons:
        status = "PASS" if comparison.invariant else "FAIL"

        print(
            f"{comparison.name:<38}"
            f"{comparison.before:>20.12e}"
            f"{comparison.after:>20.12e}"
            f"{comparison.absolute_difference:>20.12e}"
            f"{comparison.relative_difference:>20.12e}"
            f"{status:>12}"
        )

    membership_error = max_link_membership_error(transformed)

    print()
    print(f"Maximum transformed-link SU(2) numerical error: {membership_error:.12e}")

    all_invariant = all(
        comparison.invariant
        for comparison in comparisons
    )

    print()
    print("=" * 88)
    print(
        "Gauge-invariance validation status: "
        f"{'PASS' if all_invariant else 'FAIL'}"
    )

    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "gauge_invariance_validation.csv"

    rows = [
        {
            "observable": comparison.name,
            "before": comparison.before,
            "after": comparison.after,
            "absolute_difference": comparison.absolute_difference,
            "relative_difference": comparison.relative_difference,
            "invariant": comparison.invariant,
        }
        for comparison in comparisons
    ]

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved validation data: {csv_path}")

    if not all_invariant:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

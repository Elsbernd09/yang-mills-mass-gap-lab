"""
Ensemble uncertainty experiment for SU(2) lattice gauge theory.

This experiment runs multiple independent Markov chains and reports bootstrap
uncertainty estimates for:

1. Final Wilson action
2. Final average plaquette
3. Mean acceptance rate

This is a major step toward more credible numerical reporting. Instead of
showing one trajectory, we summarize an ensemble.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ymlab.ensemble import run_ensemble, summarize_bootstrap


def main() -> None:
    shape = (6, 6)
    beta = 2.0
    sweeps = 40
    epsilon = 0.18
    seeds = [2026, 2027, 2028, 2029, 2030]
    n_bootstrap = 1000

    print("Running SU(2) ensemble experiment...")
    print(f"Shape: {shape}")
    print(f"Beta: {beta}")
    print(f"Sweeps per chain: {sweeps}")
    print(f"Epsilon: {epsilon}")
    print(f"Independent chains: {len(seeds)}")
    print()

    result = run_ensemble(
        shape=shape,
        beta=beta,
        sweeps=sweeps,
        epsilon=epsilon,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
        cold_start=True,
    )

    print("Chain summaries:")
    print("-" * 80)
    print("Seed      Final Action      Final Plaquette      Mean Acceptance")
    print("-" * 80)

    for summary in result.chain_summaries:
        print(
            f"{summary.seed:<9d} "
            f"{summary.final_action:14.8f} "
            f"{summary.final_average_plaquette:20.8f} "
            f"{summary.mean_acceptance_rate:18.8f}"
        )

    print()
    print("Bootstrap uncertainty estimates:")
    print("-" * 80)
    print(summarize_bootstrap("Final Wilson action", result.action_bootstrap))
    print(summarize_bootstrap("Final average plaquette", result.plaquette_bootstrap))
    print(summarize_bootstrap("Mean acceptance rate", result.acceptance_bootstrap))

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    actions = np.asarray(
        [summary.final_action for summary in result.chain_summaries],
        dtype=float,
    )
    plaquettes = np.asarray(
        [summary.final_average_plaquette for summary in result.chain_summaries],
        dtype=float,
    )
    acceptances = np.asarray(
        [summary.mean_acceptance_rate for summary in result.chain_summaries],
        dtype=float,
    )

    chain_ids = np.arange(1, len(seeds) + 1)

    plt.figure()
    plt.scatter(chain_ids, actions)
    plt.xlabel("Independent chain")
    plt.ylabel("Final Wilson action")
    plt.title("Ensemble Final Wilson Actions")
    plt.tight_layout()
    plt.savefig(figures_dir / "ensemble_final_actions.png", dpi=200)

    plt.figure()
    plt.scatter(chain_ids, plaquettes)
    plt.xlabel("Independent chain")
    plt.ylabel("Final average plaquette")
    plt.title("Ensemble Final Average Plaquettes")
    plt.tight_layout()
    plt.savefig(figures_dir / "ensemble_final_plaquettes.png", dpi=200)

    plt.figure()
    plt.scatter(chain_ids, acceptances)
    plt.xlabel("Independent chain")
    plt.ylabel("Mean acceptance rate")
    plt.title("Ensemble Mean Acceptance Rates")
    plt.tight_layout()
    plt.savefig(figures_dir / "ensemble_acceptance_rates.png", dpi=200)

    plt.figure()
    plt.hist(result.plaquette_bootstrap.bootstrap_samples, bins=30)
    plt.xlabel("Bootstrap mean average plaquette")
    plt.ylabel("Frequency")
    plt.title("Bootstrap Distribution: Average Plaquette")
    plt.tight_layout()
    plt.savefig(figures_dir / "bootstrap_average_plaquette.png", dpi=200)

    print()
    print("Saved figures:")
    print("results/figures/ensemble_final_actions.png")
    print("results/figures/ensemble_final_plaquettes.png")
    print("results/figures/ensemble_acceptance_rates.png")
    print("results/figures/bootstrap_average_plaquette.png")


if __name__ == "__main__":
    main()

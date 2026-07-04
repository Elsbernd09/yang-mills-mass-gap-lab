"""
Baseline SU(2) lattice gauge theory experiment.

This experiment compares a cold-start lattice with a random-start lattice.
It computes:

1. Number of sites
2. Number of links
3. Number of plaquettes
4. Average plaquette
5. Wilson action
6. Action per plaquette

This is not yet a Monte Carlo simulation. It is the first validation experiment.
"""

from __future__ import annotations

from ymlab.lattice import Lattice
from ymlab.plaquette import average_plaquette, average_plaquette_action_density
from ymlab.wilson_action import wilson_action, action_per_plaquette, number_of_plaquettes


def summarize_lattice(name: str, lattice: Lattice, beta: float) -> None:
    """Print a clean summary of lattice observables."""
    print("=" * 72)
    print(name)
    print("=" * 72)
    print(f"Shape:                         {lattice.shape}")
    print(f"Dimension:                     {lattice.dim}")
    print(f"Number of sites:               {lattice.number_of_sites()}")
    print(f"Number of links:               {lattice.number_of_links()}")
    print(f"Number of plaquettes:          {number_of_plaquettes(lattice)}")
    print(f"Average plaquette:             {average_plaquette(lattice):.8f}")
    print(f"Average plaquette action:      {average_plaquette_action_density(lattice):.8f}")
    print(f"Wilson action beta={beta}:       {wilson_action(lattice, beta):.8f}")
    print(f"Action per plaquette beta={beta}: {action_per_plaquette(lattice, beta):.8f}")
    print()


def main() -> None:
    beta = 2.0

    cold_lattice = Lattice(shape=(8, 8), cold_start=True)
    random_lattice = Lattice(shape=(8, 8), cold_start=False, seed=2026)

    summarize_lattice("Cold-start SU(2) lattice", cold_lattice, beta)
    summarize_lattice("Random-start SU(2) lattice", random_lattice, beta)


if __name__ == "__main__":
    main()

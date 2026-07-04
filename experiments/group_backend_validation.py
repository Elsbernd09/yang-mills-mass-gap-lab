"""
Gauge group backend validation demo.

This experiment validates the available matrix gauge group backends used by the
project architecture layer.

It checks both SU(2) and SU(3) through the same interface:

1. Matrix dimension
2. Trace normalization
3. Identity matrix shape
4. Normalized trace of identity
5. Wilson plaquette density of identity
6. Backend validation status

This is not a full SU(N) lattice simulation. It is an architectural validation
step showing that SU(2) and SU(3) can be wrapped by a shared interface.
"""

from __future__ import annotations

from ymlab.group_interface import (
    available_groups,
    normalized_real_trace,
    validate_group_backend,
    wilson_plaquette_density_from_matrix,
)


def main() -> None:
    print("Gauge Group Backend Validation")
    print("=" * 72)
    print()

    for group in available_groups():
        identity = group.identity()

        valid = validate_group_backend(
            group=group,
            samples=10,
            seed=2026,
        )

        normalized_trace = normalized_real_trace(
            group=group,
            matrix=identity,
        )

        wilson_density = wilson_plaquette_density_from_matrix(
            group=group,
            plaquette_matrix=identity,
        )

        print(f"Group: {group.name}")
        print("-" * 72)
        print(f"Matrix dimension:              {group.dimension}")
        print(f"Trace normalization:           {group.trace_normalization:.8f}")
        print(f"Identity shape:                {identity.shape}")
        print(f"Backend validation passed:     {valid}")
        print(f"Normalized trace of identity:  {normalized_trace:.8f}")
        print(f"Wilson density of identity:    {wilson_density:.8f}")
        print()

    print("=" * 72)
    print("Group backend validation complete.")


if __name__ == "__main__":
    main()

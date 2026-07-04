"""
Validation suite for the Yang-Mills Mass Gap Laboratory.

This script runs internal consistency checks across the project:

1. SU(2) identity and random matrices are valid.
2. Cold-start lattice has plaquette = identity.
3. Cold-start Wilson action is zero.
4. Cold-start Wilson loops equal one.
5. Metropolis updates preserve SU(2) link structure.
6. Creutz ratio recovers exact synthetic area-law string tension.
7. Dimension-dependent lattice counts match expected formulas.
8. SU(3) Gell-Mann matrices are Hermitian, traceless, and normalized.

This is not a proof of Yang-Mills theory. It is a validation suite for the
finite-lattice computational framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import traceback

import numpy as np

from ymlab.covariance_fits import (
    fit_correlated_periodic_cosh,
    regularized_inverse_covariance,
)
from ymlab.creutz import creutz_ratio_from_values
from ymlab.gauge_transformations import (
    gauge_transform_lattice,
    random_gauge_field,
)
from ymlab.glueball import (
    ensemble_connected_correlator,
    scalar_glueball_time_series,
)
from ymlab.dimensional_analysis import theoretical_plaquettes_per_site
from ymlab.lattice import Lattice
from ymlab.monte_carlo import metropolis_sweep
from ymlab.observables import rectangular_wilson_loop
from ymlab.plaquette import average_plaquette, plaquette
from ymlab.resampling import (
    delete_one_block_jackknife_mean,
    moving_block_bootstrap_mean,
)
from ymlab.spectroscopy import (
    arccosh_effective_mass,
    fit_periodic_cosh,
    periodic_cosh_correlator,
)
from ymlab.smearing import (
    maximum_su2_link_error,
    smear_spatial_links,
)
from ymlab.operator_basis import (
    create_smearing_basis,
    ensemble_correlator_matrix,
    measure_operator_basis,
)
from ymlab.su2 import identity as su2_identity
from ymlab.su2 import is_su2, random_su2
from ymlab.su3 import (
    gell_mann_matrices,
    identity as su3_identity,
    is_hermitian,
    is_su3,
    is_traceless,
    random_su3,
)
from ymlab.variational import (
    principal_log_effective_masses,
    solve_regularized_gevp,
)
from ymlab.validation import compare_action_differences
from ymlab.wilson_action import number_of_plaquettes, wilson_action


@dataclass
class ValidationCheck:
    """Single validation check result."""

    name: str
    passed: bool
    details: str


def run_check(name: str, function) -> ValidationCheck:
    """Run one validation check safely."""
    try:
        details = function()
        return ValidationCheck(name=name, passed=True, details=details)
    except Exception as error:
        return ValidationCheck(
            name=name,
            passed=False,
            details=f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
        )


def check_su2_identity_and_random() -> str:
    rng = np.random.default_rng(2026)

    assert is_su2(su2_identity())

    for _ in range(20):
        u = random_su2(rng)
        assert is_su2(u)

    return "SU(2) identity and 20 random SU(2) samples validated."


def check_cold_start_plaquette() -> str:
    lattice = Lattice(shape=(6, 6), cold_start=True)

    up = plaquette(lattice, site=(0, 0), mu=0, nu=1)

    assert np.allclose(up, su2_identity())
    assert np.isclose(average_plaquette(lattice), 1.0)

    return "Cold-start plaquette equals identity and average plaquette equals 1."


def check_cold_start_wilson_action() -> str:
    lattice = Lattice(shape=(6, 6), cold_start=True)

    action = wilson_action(lattice, beta=2.0)

    assert np.isclose(action, 0.0)

    return "Cold-start Wilson action equals 0."


def check_cold_start_wilson_loop() -> str:
    lattice = Lattice(shape=(6, 6), cold_start=True)

    for width in [1, 2, 3]:
        for height in [1, 2, 3]:
            value = rectangular_wilson_loop(
                lattice=lattice,
                site=(0, 0),
                mu=0,
                nu=1,
                width=width,
                height=height,
            )
            assert np.isclose(value, 1.0)

    return "Cold-start rectangular Wilson loops up to 3x3 equal 1."


def check_metropolis_preserves_su2() -> str:
    lattice = Lattice(shape=(4, 4), cold_start=True, seed=2026)

    for _ in range(5):
        metropolis_sweep(lattice, beta=2.0, epsilon=0.15)

    for site in lattice.sites():
        for mu in range(lattice.dim):
            assert is_su2(lattice.get_link(site, mu), atol=1e-8)

    return "Five Metropolis sweeps preserved SU(2) link membership."


def check_creutz_exact_area_law() -> str:
    sigma = 0.35

    def w(width: int, height: int) -> float:
        return float(np.exp(-sigma * width * height))

    result = creutz_ratio_from_values(
        w_rt=w(2, 2),
        w_r1_t1=w(1, 1),
        w_rt1=w(2, 1),
        w_r1_t=w(1, 2),
        width=2,
        height=2,
    )

    assert result.valid
    assert np.isclose(result.value, sigma)

    return f"Creutz ratio recovered exact synthetic area-law sigma={sigma}."


def check_dimension_counts() -> str:
    shapes = [
        (6, 6),
        (4, 4, 4),
        (3, 3, 3, 3),
    ]

    for shape in shapes:
        lattice = Lattice(shape=shape, cold_start=True)
        dimension = len(shape)
        sites = lattice.number_of_sites()
        expected_links = sites * dimension
        expected_plaquettes = sites * comb(dimension, 2)

        assert lattice.number_of_links() == expected_links
        assert number_of_plaquettes(lattice) == expected_plaquettes
        assert theoretical_plaquettes_per_site(dimension) == comb(dimension, 2)

    return "2D, 3D, and 4D lattice site/link/plaquette counts validated."


def check_su3_identity_and_random() -> str:
    rng = np.random.default_rng(2026)

    assert is_su3(su3_identity())

    for _ in range(10):
        u = random_su3(rng)
        assert is_su3(u, atol=1e-8)

    return "SU(3) identity and 10 random SU(3) samples validated."


def check_gell_mann_matrices() -> str:
    matrices = gell_mann_matrices()

    assert len(matrices) == 8

    for matrix in matrices:
        assert is_hermitian(matrix)
        assert is_traceless(matrix)

    for i, a in enumerate(matrices):
        for j, b in enumerate(matrices):
            trace_value = np.trace(a @ b)

            if i == j:
                assert np.isclose(trace_value, 2.0)
            else:
                assert np.isclose(trace_value, 0.0)

    return "Gell-Mann matrices are Hermitian, traceless, and normalized."



def check_gauge_invariance() -> str:
    lattice = Lattice(
        shape=(4, 4),
        cold_start=True,
        seed=2026,
    )

    for _ in range(4):
        metropolis_sweep(
            lattice,
            beta=2.0,
            epsilon=0.15,
        )

    gauge_field = random_gauge_field(
        lattice,
        seed=314159,
    )

    transformed = gauge_transform_lattice(
        lattice,
        gauge_field,
    )

    action_before = wilson_action(
        lattice,
        beta=2.0,
    )
    action_after = wilson_action(
        transformed,
        beta=2.0,
    )

    plaquette_before = average_plaquette(lattice)
    plaquette_after = average_plaquette(transformed)

    loop_before = rectangular_wilson_loop(
        lattice=lattice,
        site=(0, 0),
        mu=0,
        nu=1,
        width=2,
        height=2,
    )
    loop_after = rectangular_wilson_loop(
        lattice=transformed,
        site=(0, 0),
        mu=0,
        nu=1,
        width=2,
        height=2,
    )

    assert np.isclose(
        action_before,
        action_after,
        atol=1e-8,
        rtol=1e-8,
    )
    assert np.isclose(
        plaquette_before,
        plaquette_after,
        atol=1e-8,
        rtol=1e-8,
    )
    assert np.isclose(
        loop_before,
        loop_after,
        atol=1e-8,
        rtol=1e-8,
    )

    return (
        "Random local gauge transformation preserved Wilson action, "
        "average plaquette, and a closed Wilson loop."
    )


def check_correlation_aware_resampling() -> str:
    values = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ],
        dtype=float,
    )

    bootstrap = moving_block_bootstrap_mean(
        values=values,
        block_size=2,
        n_bootstrap=200,
        seed=2026,
    )

    jackknife = delete_one_block_jackknife_mean(
        values=values,
        block_size=2,
    )

    assert np.isclose(
        bootstrap.estimate,
        np.mean(values),
    )
    assert bootstrap.standard_error >= 0.0
    assert np.isclose(
        jackknife.estimate,
        np.mean(values),
    )
    assert jackknife.standard_error >= 0.0

    return (
        "Moving-block bootstrap and block jackknife "
        "returned finite mean uncertainty estimates."
    )


def check_glueball_operator_pipeline() -> str:
    lattice = Lattice(
        shape=(4, 3, 3),
        cold_start=True,
        seed=2026,
    )

    ensemble = []

    for _ in range(3):
        metropolis_sweep(
            lattice=lattice,
            beta=2.0,
            epsilon=0.15,
        )

        ensemble.append(
            scalar_glueball_time_series(
                lattice=lattice,
                time_direction=0,
            )
        )

    ensemble = np.asarray(
        ensemble,
        dtype=float,
    )

    result = ensemble_connected_correlator(
        ensemble
    )

    assert result.number_of_configurations == 3
    assert result.temporal_extent == 4
    assert result.correlation.shape == (4,)
    assert np.all(
        np.isfinite(result.correlation)
    )

    return (
        "Gauge-invariant scalar operator ensemble and "
        "connected Euclidean correlator validated."
    )


def check_periodic_spectroscopy() -> str:
    temporal_extent = 10
    exact_mass = 0.625

    correlation = periodic_cosh_correlator(
        lag=np.arange(
            temporal_extent,
            dtype=float,
        ),
        amplitude=1.5,
        mass=exact_mass,
        temporal_extent=temporal_extent,
    )

    effective_mass = arccosh_effective_mass(
        correlation
    )

    finite_mass = effective_mass[
        np.isfinite(effective_mass)
    ]

    assert len(finite_mass) > 0

    assert np.allclose(
        finite_mass,
        exact_mass,
        atol=1e-10,
        rtol=1e-10,
    )

    fit = fit_periodic_cosh(
        correlation=correlation,
        fit_start=1,
        fit_stop=5,
    )

    assert fit.success
    assert np.isclose(
        fit.mass,
        exact_mass,
        atol=1e-7,
    )

    return (
        "Periodic cosh fit and arccosh effective-mass "
        "estimator recovered an exact synthetic mass."
    )


def check_local_global_action_consistency() -> str:
    from ymlab.su2 import small_random_su2

    lattice = Lattice(
        shape=(3, 3, 3),
        cold_start=False,
        seed=2026,
    )

    rng = np.random.default_rng(314159)

    proposals = [
        ((0, 0, 0), 0),
        ((1, 2, 0), 1),
        ((2, 1, 2), 2),
    ]

    for site, mu in proposals:
        old_link = lattice.get_link(
            site,
            mu,
        )

        proposal = (
            small_random_su2(
                epsilon=0.15,
                rng=rng,
            )
            @ old_link
        )

        comparison = compare_action_differences(
            lattice=lattice,
            site=site,
            mu=mu,
            proposal=proposal,
            beta=2.0,
            atol=1e-10,
            rtol=1e-10,
        )

        assert comparison.consistent
        assert comparison.absolute_error < 1e-9

    return (
        "Local staple action differences matched full "
        "Wilson-action differences on random 3D proposals."
    )


def check_spatial_smearing_pipeline() -> str:
    lattice = Lattice(
        shape=(4, 3, 3),
        cold_start=True,
        seed=2026,
    )

    for _ in range(3):
        metropolis_sweep(
            lattice=lattice,
            beta=2.0,
            epsilon=0.15,
        )

    gauge_field = random_gauge_field(
        lattice=lattice,
        seed=314159,
    )

    transformed = gauge_transform_lattice(
        lattice=lattice,
        gauge_field=gauge_field,
    )

    smeared = smear_spatial_links(
        lattice=lattice,
        time_direction=0,
        alpha=0.5,
        steps=2,
    )

    transformed_smeared = smear_spatial_links(
        lattice=transformed,
        time_direction=0,
        alpha=0.5,
        steps=2,
    )

    operator_before = scalar_glueball_time_series(
        lattice=smeared,
        time_direction=0,
    )

    operator_after = scalar_glueball_time_series(
        lattice=transformed_smeared,
        time_direction=0,
    )

    assert np.allclose(
        operator_before,
        operator_after,
        atol=1e-8,
        rtol=1e-8,
    )

    assert maximum_su2_link_error(
        smeared
    ) < 1e-8

    return (
        "Spatial smearing preserved SU(2) numerically "
        "and retained gauge-invariant scalar operators."
    )


def check_operator_correlator_matrix() -> str:
    lattice = Lattice(
        shape=(4, 3, 3),
        cold_start=True,
        seed=2026,
    )

    basis = create_smearing_basis(
        smearing_levels=[
            0,
            2,
        ],
        alpha=0.5,
        time_direction=0,
    )

    ensemble = []

    for _ in range(3):
        metropolis_sweep(
            lattice=lattice,
            beta=2.0,
            epsilon=0.15,
        )

        ensemble.append(
            measure_operator_basis(
                lattice=lattice,
                basis=basis,
            )
        )

    ensemble = np.asarray(
        ensemble,
        dtype=float,
    )

    result = ensemble_correlator_matrix(
        ensemble
    )

    assert (
        result.correlation_matrices.shape
        == (4, 2, 2)
    )

    assert np.allclose(
        result.correlation_matrices[0],
        result.correlation_matrices[0].T,
        atol=1e-10,
        rtol=1e-10,
    )

    return (
        "Multi-operator scalar basis produced a "
        "finite connected Euclidean correlator matrix."
    )


def check_generalized_eigenvalue_pipeline() -> str:
    energies = np.array(
        [
            0.4,
            0.9,
        ],
        dtype=float,
    )

    mixing = np.array(
        [
            [1.0, 0.3],
            [0.2, 1.1],
        ],
        dtype=float,
    )

    matrices = []

    for lag in range(6):
        diagonal = np.diag(
            np.exp(
                -energies * lag
            )
        )

        matrices.append(
            mixing
            @ diagonal
            @ mixing.T
        )

    matrices = np.asarray(
        matrices,
        dtype=float,
    )

    result = solve_regularized_gevp(
        correlation_matrices=matrices,
        reference_time=1,
        track_states=False,
    )

    masses = principal_log_effective_masses(
        result.principal_correlators
    )

    assert result.retained_rank == 2

    assert np.allclose(
        result.principal_correlators[
            1
        ],
        1.0,
        atol=1e-10,
        rtol=1e-10,
    )

    assert np.allclose(
        masses[
            1:-1,
            0,
        ],
        energies[0],
        atol=1e-8,
        rtol=1e-8,
    )

    assert np.allclose(
        masses[
            1:-1,
            1,
        ],
        energies[1],
        atol=1e-8,
        rtol=1e-8,
    )

    return (
        "Regularized GEVP recovered exact synthetic "
        "principal correlators and variational masses."
    )


def check_correlated_spectroscopy_fit() -> str:
    temporal_extent = 10
    exact_amplitude = 1.4
    exact_mass = 0.57

    correlation = periodic_cosh_correlator(
        lag=np.arange(
            temporal_extent,
            dtype=float,
        ),
        amplitude=exact_amplitude,
        mass=exact_mass,
        temporal_extent=temporal_extent,
    )

    covariance = np.full(
        (
            temporal_extent,
            temporal_extent,
        ),
        1e-5,
        dtype=float,
    )

    covariance += np.eye(
        temporal_extent,
        dtype=float,
    ) * 9e-5

    inverse_result = (
        regularized_inverse_covariance(
            covariance=covariance,
            relative_cutoff=1e-12,
        )
    )

    assert inverse_result.retained_rank == temporal_extent

    fit = fit_correlated_periodic_cosh(
        correlation=correlation,
        covariance=covariance,
        fit_start=1,
        fit_stop=5,
        shrinkage=0.05,
        relative_cutoff=1e-12,
    )

    assert fit.success

    assert np.isclose(
        fit.mass,
        exact_mass,
        atol=1e-5,
        rtol=1e-5,
    )

    assert np.isclose(
        fit.amplitude,
        exact_amplitude,
        atol=1e-5,
        rtol=1e-5,
    )

    return (
        "Regularized inverse covariance and correlated "
        "periodic-cosh fit recovered exact synthetic parameters."
    )

def main() -> None:
    checks = [
        ("SU(2) identity/random validation", check_su2_identity_and_random),
        ("Cold-start plaquette validation", check_cold_start_plaquette),
        ("Cold-start Wilson action validation", check_cold_start_wilson_action),
        ("Cold-start Wilson loop validation", check_cold_start_wilson_loop),
        ("Metropolis SU(2) preservation validation", check_metropolis_preserves_su2),
        ("Creutz exact area-law validation", check_creutz_exact_area_law),
        ("Dimension count validation", check_dimension_counts),
        ("SU(3) identity/random validation", check_su3_identity_and_random),
        ("Gell-Mann basis validation", check_gell_mann_matrices),
        ("Local gauge-invariance validation", check_gauge_invariance),
        ("Correlation-aware resampling validation", check_correlation_aware_resampling),
        ("Scalar glueball-style correlator validation", check_glueball_operator_pipeline),
        ("Periodic spectroscopy validation", check_periodic_spectroscopy),
        ("Local/global Wilson-action consistency", check_local_global_action_consistency),
        ("Spatial smearing pipeline validation", check_spatial_smearing_pipeline),
        ("Operator correlator matrix validation", check_operator_correlator_matrix),
        ("Generalized eigenvalue validation", check_generalized_eigenvalue_pipeline),
        ("Correlated spectroscopy fit validation", check_correlated_spectroscopy_fit),
    ]

    results = [run_check(name, function) for name, function in checks]

    print("Yang-Mills Mass Gap Laboratory Validation Suite")
    print("=" * 72)
    print()

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        print(f"       {result.details.splitlines()[0]}")
        print()

    passed = sum(result.passed for result in results)
    total = len(results)

    print("=" * 72)
    print(f"Validation checks passed: {passed}/{total}")

    if passed != total:
        print()
        print("Failure details:")
        print("-" * 72)

        for result in results:
            if not result.passed:
                print(f"{result.name}")
                print(result.details)
                print("-" * 72)

        raise SystemExit(1)

    print("All validation checks passed.")


if __name__ == "__main__":
    main()

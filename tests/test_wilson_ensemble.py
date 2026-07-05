import numpy as np
import pytest

from ymlab.lattice import Lattice
from ymlab.wilson_ensemble import (
    EnsembleCreutzBootstrapResult,
    WilsonLoopBasis,
    block_bootstrap_creutz_ratios,
    create_rectangular_loop_basis,
    creutz_ratios_from_loop_means,
    measure_wilson_loop_basis,
    required_creutz_shapes,
    square_creutz_plateau_values,
)


def synthetic_area_law_ensemble(
    sigma=0.35,
    configurations=100,
    maximum_width=4,
    maximum_height=4,
    noise_scale=0.002,
    seed=123,
):
    basis = create_rectangular_loop_basis(
        maximum_width=maximum_width,
        maximum_height=maximum_height,
    )

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for _ in range(
        configurations
    ):
        common_fluctuation = rng.normal(
            loc=0.0,
            scale=noise_scale,
        )

        row = []

        for width, height in basis.shapes:
            exact = np.exp(
                -sigma
                * width
                * height
            )

            multiplicative = np.exp(
                common_fluctuation
                + rng.normal(
                    loc=0.0,
                    scale=noise_scale,
                )
            )

            row.append(
                exact
                * multiplicative
            )

        rows.append(
            row
        )

    return (
        np.asarray(
            rows,
            dtype=float,
        ),
        basis,
    )


def test_rectangular_loop_basis_shape_count():
    basis = create_rectangular_loop_basis(
        maximum_width=3,
        maximum_height=4,
    )

    assert len(
        basis.shapes
    ) == 12

    assert (
        3,
        4,
    ) in basis.shapes


def test_required_creutz_shapes():
    basis = create_rectangular_loop_basis(
        maximum_width=3,
        maximum_height=3,
    )

    assert required_creutz_shapes(
        basis
    ) == (
        (
            2,
            2,
        ),
        (
            2,
            3,
        ),
        (
            3,
            2,
        ),
        (
            3,
            3,
        ),
    )


def test_cold_lattice_loop_basis_is_one():
    lattice = Lattice(
        shape=(
            6,
            6,
        ),
        cold_start=True,
    )

    basis = create_rectangular_loop_basis(
        maximum_width=3,
        maximum_height=3,
    )

    values = measure_wilson_loop_basis(
        lattice=lattice,
        basis=basis,
        mu=0,
        nu=1,
    )

    assert np.allclose(
        values,
        1.0,
        atol=1e-12,
        rtol=1e-12,
    )


def test_exact_area_law_creutz_ratios_recover_sigma():
    sigma = 0.42

    basis = create_rectangular_loop_basis(
        maximum_width=4,
        maximum_height=4,
    )

    loop_means = np.asarray(
        [
            np.exp(
                -sigma
                * width
                * height
            )
            for width, height in basis.shapes
        ],
        dtype=float,
    )

    values = creutz_ratios_from_loop_means(
        loop_means=loop_means,
        basis=basis,
    )

    assert np.allclose(
        values,
        sigma,
        atol=1e-12,
        rtol=1e-12,
    )


def test_block_bootstrap_recovers_synthetic_area_law():
    sigma = 0.35

    ensemble, basis = (
        synthetic_area_law_ensemble(
            sigma=sigma,
            configurations=120,
            seed=123,
        )
    )

    result = block_bootstrap_creutz_ratios(
        loop_ensemble=ensemble,
        basis=basis,
        n_bootstrap=100,
        block_size=4,
        seed=321,
    )

    assert isinstance(
        result,
        EnsembleCreutzBootstrapResult,
    )

    assert len(
        result.creutz_points
    ) > 0

    for point in result.creutz_points:
        assert point.valid_fraction > 0.95

        assert abs(
            point.bootstrap_mean
            - sigma
        ) < 0.03


def test_block_bootstrap_preserves_joint_loop_columns():
    sigma = 0.25

    ensemble, basis = (
        synthetic_area_law_ensemble(
            sigma=sigma,
            configurations=50,
            seed=456,
        )
    )

    result = block_bootstrap_creutz_ratios(
        loop_ensemble=ensemble,
        basis=basis,
        n_bootstrap=30,
        block_size=5,
        seed=654,
    )

    assert (
        result.loop_mean_samples.shape
        == (
            30,
            len(
                basis.shapes
            ),
        )
    )

    assert (
        result.creutz_samples.shape[0]
        == 30
    )


def test_invalid_creutz_ratio_remains_nan():
    basis = create_rectangular_loop_basis(
        maximum_width=2,
        maximum_height=2,
    )

    loop_means = np.ones(
        len(
            basis.shapes
        ),
        dtype=float,
    )

    loop_means[
        basis.index(
            (
                2,
                2,
            )
        )
    ] = -1.0

    values = creutz_ratios_from_loop_means(
        loop_means=loop_means,
        basis=basis,
    )

    assert np.isnan(
        values[
            0
        ]
    )


def test_square_plateau_filter():
    ensemble, basis = (
        synthetic_area_law_ensemble(
            sigma=0.3,
            configurations=80,
            maximum_width=4,
            maximum_height=4,
            seed=789,
        )
    )

    result = block_bootstrap_creutz_ratios(
        loop_ensemble=ensemble,
        basis=basis,
        n_bootstrap=50,
        block_size=4,
        seed=987,
    )

    values = square_creutz_plateau_values(
        result,
        minimum_valid_fraction=0.8,
    )

    assert len(
        values
    ) == 3

    assert np.all(
        np.isfinite(
            values
        )
    )


def test_duplicate_loop_shapes_rejected():
    with pytest.raises(ValueError):
        WilsonLoopBasis(
            shapes=(
                (
                    1,
                    1,
                ),
                (
                    1,
                    1,
                ),
            )
        )


def test_invalid_block_size_rejected():
    ensemble, basis = (
        synthetic_area_law_ensemble(
            configurations=20,
        )
    )

    with pytest.raises(ValueError):
        block_bootstrap_creutz_ratios(
            loop_ensemble=ensemble,
            basis=basis,
            n_bootstrap=20,
            block_size=21,
            seed=123,
        )

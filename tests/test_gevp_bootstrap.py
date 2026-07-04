import numpy as np
import pytest

from ymlab.gevp_bootstrap import (
    BootstrapInterval,
    GEVPBootstrapResult,
    bootstrap_configuration_indices,
    bootstrap_gevp,
    circular_block_bootstrap_indices,
    match_states_to_reference,
    metric_overlap_matrix,
    summarize_bootstrap_distribution,
)


def synthetic_operator_ensemble(
    number_of_configurations=60,
    temporal_extent=8,
    seed=123,
):
    rng = np.random.default_rng(
        seed
    )

    energies = np.array(
        [
            0.35,
            0.80,
        ],
        dtype=float,
    )

    mixing = np.array(
        [
            [1.0, 0.35],
            [0.25, 1.1],
        ],
        dtype=float,
    )

    ensemble = []

    for _ in range(
        number_of_configurations
    ):
        state_amplitudes = rng.normal(
            loc=0.0,
            scale=1.0,
            size=2,
        )

        time_series = np.zeros(
            (
                2,
                temporal_extent,
            ),
            dtype=float,
        )

        for time in range(
            temporal_extent
        ):
            decayed_state = (
                state_amplitudes
                * np.exp(
                    -0.5
                    * energies
                    * time
                )
            )

            signal = (
                mixing
                @ decayed_state
            )

            noise = rng.normal(
                scale=0.02,
                size=2,
            )

            time_series[
                :,
                time,
            ] = signal + noise

        ensemble.append(
            time_series
        )

    return np.asarray(
        ensemble,
        dtype=float,
    )


def test_circular_block_indices_shape_and_bounds():
    rng = np.random.default_rng(
        123
    )

    indices = circular_block_bootstrap_indices(
        number_of_configurations=20,
        block_size=4,
        rng=rng,
    )

    assert indices.shape == (
        20,
    )

    assert np.all(
        indices >= 0
    )

    assert np.all(
        indices < 20
    )


def test_iid_bootstrap_indices_shape():
    rng = np.random.default_rng(
        123
    )

    indices = bootstrap_configuration_indices(
        number_of_configurations=25,
        rng=rng,
        mode="iid",
    )

    assert indices.shape == (
        25,
    )


def test_invalid_bootstrap_mode_rejected():
    rng = np.random.default_rng(
        123
    )

    with pytest.raises(ValueError):
        bootstrap_configuration_indices(
            number_of_configurations=20,
            rng=rng,
            mode="invalid",
        )


def test_metric_overlap_identity_basis():
    reference = np.eye(
        2
    )

    candidate = np.eye(
        2
    )

    metric = np.eye(
        2
    )

    overlap = metric_overlap_matrix(
        reference_vectors=reference,
        candidate_vectors=candidate,
        metric_matrix=metric,
    )

    assert np.allclose(
        overlap,
        np.eye(
            2
        ),
    )


def test_state_matching_recovers_permutation_and_sign():
    reference = np.eye(
        2
    )

    candidate = np.array(
        [
            [0.0, -1.0],
            [1.0, 0.0],
        ]
    )

    candidate_values = np.array(
        [
            20.0,
            10.0,
        ]
    )

    match = match_states_to_reference(
        reference_vectors=reference,
        candidate_vectors=candidate,
        candidate_values=candidate_values,
        metric_matrix=np.eye(
            2
        ),
    )

    assert np.array_equal(
        match.permutation,
        np.array(
            [
                1,
                0,
            ]
        ),
    )

    assert np.allclose(
        match.matched_values,
        np.array(
            [
                10.0,
                20.0,
            ]
        ),
    )

    assert np.allclose(
        match.matched_vectors,
        reference,
    )

    assert np.allclose(
        match.matched_overlaps,
        1.0,
    )


def test_metric_overlap_respects_nondiagonal_metric():
    reference = np.eye(
        2
    )

    candidate = np.eye(
        2
    )

    metric = np.array(
        [
            [2.0, 0.25],
            [0.25, 1.0],
        ]
    )

    overlap = metric_overlap_matrix(
        reference_vectors=reference,
        candidate_vectors=candidate,
        metric_matrix=metric,
    )

    assert np.isclose(
        overlap[0, 0],
        1.0,
    )

    assert np.isclose(
        overlap[1, 1],
        1.0,
    )

    assert overlap[0, 1] > 0.0


def test_bootstrap_gevp_returns_accepted_samples():
    ensemble = synthetic_operator_ensemble(
        number_of_configurations=70,
        temporal_extent=8,
        seed=123,
    )

    result = bootstrap_gevp(
        operator_ensemble=ensemble,
        n_bootstrap=30,
        reference_time=0,
        relative_cutoff=1e-8,
        mode="iid",
        seed=321,
    )

    assert isinstance(
        result,
        GEVPBootstrapResult,
    )

    assert result.accepted_replicates > 0

    assert (
        result.principal_samples.shape[0]
        == result.accepted_replicates
    )

    assert (
        result.principal_samples.shape[2]
        == result.central_rank
    )

    assert np.all(
        np.isfinite(
            result.principal_samples
        )
    )


def test_block_bootstrap_gevp_runs():
    ensemble = synthetic_operator_ensemble(
        number_of_configurations=60,
        temporal_extent=8,
        seed=456,
    )

    result = bootstrap_gevp(
        operator_ensemble=ensemble,
        n_bootstrap=20,
        reference_time=0,
        relative_cutoff=1e-8,
        mode="circular_block",
        block_size=4,
        seed=654,
    )

    assert result.accepted_replicates > 0

    assert result.bootstrap_mode == (
        "circular_block"
    )

    assert result.block_size == 4


def test_summarize_bootstrap_distribution():
    samples = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, np.nan],
            ],
            [
                [2.0, 3.0],
                [4.0, 5.0],
            ],
            [
                [3.0, 4.0],
                [5.0, 6.0],
            ],
        ]
    )

    summary = (
        summarize_bootstrap_distribution(
            samples,
            confidence_level=0.95,
        )
    )

    assert isinstance(
        summary,
        BootstrapInterval,
    )

    assert summary.mean.shape == (
        2,
        2,
    )

    assert summary.finite_counts[
        1,
        1,
    ] == 2

    assert np.isclose(
        summary.mean[
            0,
            0,
        ],
        2.0,
    )


def test_summary_rejects_invalid_confidence_level():
    samples = np.ones(
        (
            10,
            3,
        )
    )

    with pytest.raises(ValueError):
        summarize_bootstrap_distribution(
            samples,
            confidence_level=1.0,
        )

import numpy as np
import pytest

from ymlab.ensemble import (
    BootstrapResult,
    ChainSummary,
    EnsembleResult,
    bootstrap_mean,
    run_independent_chain,
    run_ensemble,
    summarize_bootstrap,
)


def test_bootstrap_mean_returns_result():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    result = bootstrap_mean(values, n_bootstrap=100, seed=123)

    assert isinstance(result, BootstrapResult)
    assert np.isclose(result.mean, 2.5)
    assert result.standard_error >= 0.0
    assert len(result.bootstrap_samples) == 100


def test_bootstrap_rejects_empty_values():
    with pytest.raises(ValueError):
        bootstrap_mean(np.array([]))


def test_bootstrap_rejects_bad_confidence_level():
    with pytest.raises(ValueError):
        bootstrap_mean(np.array([1.0, 2.0]), confidence_level=1.5)


def test_run_independent_chain_returns_summary():
    summary = run_independent_chain(
        shape=(3, 3),
        beta=2.0,
        sweeps=2,
        epsilon=0.1,
        seed=123,
    )

    assert isinstance(summary, ChainSummary)
    assert np.isfinite(summary.final_action)
    assert np.isfinite(summary.final_average_plaquette)
    assert 0.0 <= summary.mean_acceptance_rate <= 1.0


def test_run_ensemble_returns_result():
    result = run_ensemble(
        shape=(3, 3),
        beta=2.0,
        sweeps=2,
        epsilon=0.1,
        seeds=[1, 2, 3],
        n_bootstrap=50,
    )

    assert isinstance(result, EnsembleResult)
    assert len(result.chain_summaries) == 3
    assert isinstance(result.action_bootstrap, BootstrapResult)
    assert isinstance(result.plaquette_bootstrap, BootstrapResult)
    assert isinstance(result.acceptance_bootstrap, BootstrapResult)


def test_run_ensemble_rejects_empty_seeds():
    with pytest.raises(ValueError):
        run_ensemble(
            shape=(3, 3),
            beta=2.0,
            sweeps=2,
            epsilon=0.1,
            seeds=[],
        )


def test_summarize_bootstrap_contains_name():
    result = bootstrap_mean(np.array([1.0, 2.0, 3.0]), n_bootstrap=50, seed=123)

    text = summarize_bootstrap("Test Observable", result)

    assert "Test Observable" in text
    assert "mean=" in text
    assert "SE=" in text

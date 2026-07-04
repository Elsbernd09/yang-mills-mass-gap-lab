import numpy as np
import pytest

from ymlab.scans import (
    ScanPoint,
    ScanResult,
    run_scan_point,
    run_beta_scan,
    run_finite_size_scan,
    run_beta_size_grid,
    scan_result_table,
)


def test_run_scan_point_returns_scan_point():
    point = run_scan_point(
        shape=(3, 3),
        beta=2.0,
        sweeps=2,
        epsilon=0.1,
        seeds=[1, 2],
        n_bootstrap=20,
    )

    assert isinstance(point, ScanPoint)
    assert point.shape == (3, 3)
    assert point.beta == 2.0
    assert np.isfinite(point.plaquette.mean)


def test_run_beta_scan_returns_result():
    result = run_beta_scan(
        shape=(3, 3),
        betas=[1.5, 2.0],
        sweeps=2,
        epsilon=0.1,
        seeds=[1, 2],
        n_bootstrap=20,
    )

    assert isinstance(result, ScanResult)
    assert len(result.points) == 2


def test_run_finite_size_scan_returns_result():
    result = run_finite_size_scan(
        shapes=[(3, 3), (4, 4)],
        beta=2.0,
        sweeps=2,
        epsilon=0.1,
        seeds=[1, 2],
        n_bootstrap=20,
    )

    assert isinstance(result, ScanResult)
    assert len(result.points) == 2


def test_run_beta_size_grid_returns_expected_count():
    result = run_beta_size_grid(
        shapes=[(3, 3), (4, 4)],
        betas=[1.5, 2.0],
        sweeps=2,
        epsilon=0.1,
        seeds=[1, 2],
        n_bootstrap=20,
    )

    assert len(result.points) == 4


def test_scan_result_table_has_expected_keys():
    result = run_beta_scan(
        shape=(3, 3),
        betas=[2.0],
        sweeps=2,
        epsilon=0.1,
        seeds=[1, 2],
        n_bootstrap=20,
    )

    table = scan_result_table(result)
    row = table[0]

    assert "shape" in row
    assert "volume" in row
    assert "beta" in row
    assert "plaquette_mean" in row
    assert "plaquette_se" in row


def test_beta_scan_rejects_empty_betas():
    with pytest.raises(ValueError):
        run_beta_scan(
            shape=(3, 3),
            betas=[],
            sweeps=2,
            epsilon=0.1,
            seeds=[1, 2],
        )


def test_finite_size_scan_rejects_empty_shapes():
    with pytest.raises(ValueError):
        run_finite_size_scan(
            shapes=[],
            beta=2.0,
            sweeps=2,
            epsilon=0.1,
            seeds=[1, 2],
        )

import numpy as np

from ymlab.group_interface import (
    MatrixGaugeGroup,
    su2_group,
    su3_group,
    normalized_real_trace,
    wilson_plaquette_density_from_matrix,
    validate_group_backend,
    available_groups,
)


def test_su2_group_interface():
    group = su2_group()

    assert isinstance(group, MatrixGaugeGroup)
    assert group.name == "SU(2)"
    assert group.dimension == 2
    assert np.isclose(group.trace_normalization, 0.5)
    assert group.is_member(group.identity())


def test_su3_group_interface():
    group = su3_group()

    assert isinstance(group, MatrixGaugeGroup)
    assert group.name == "SU(3)"
    assert group.dimension == 3
    assert np.isclose(group.trace_normalization, 1.0 / 3.0)
    assert group.is_member(group.identity())


def test_normalized_real_trace_identity_su2():
    group = su2_group()

    assert np.isclose(normalized_real_trace(group, group.identity()), 1.0)


def test_normalized_real_trace_identity_su3():
    group = su3_group()

    assert np.isclose(normalized_real_trace(group, group.identity()), 1.0)


def test_wilson_density_identity_su2():
    group = su2_group()

    density = wilson_plaquette_density_from_matrix(group, group.identity())

    assert np.isclose(density, 0.0)


def test_wilson_density_identity_su3():
    group = su3_group()

    density = wilson_plaquette_density_from_matrix(group, group.identity())

    assert np.isclose(density, 0.0)


def test_validate_su2_backend():
    assert validate_group_backend(su2_group(), samples=5, seed=123)


def test_validate_su3_backend():
    assert validate_group_backend(su3_group(), samples=5, seed=123)


def test_available_groups():
    groups = available_groups()
    names = [group.name for group in groups]

    assert "SU(2)" in names
    assert "SU(3)" in names

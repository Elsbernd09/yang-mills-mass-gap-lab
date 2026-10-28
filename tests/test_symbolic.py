import numpy as np
import sympy as sp

from ymlab import symbolic as S
from ymlab.diagnostics import effective_sample_size, integrated_autocorrelation_time
from ymlab.spectroscopy import arccosh_effective_mass, periodic_cosh_correlator


def test_all_models_render_latex():
    for r in S.all_models():
        assert r.latex and "\\" in r.latex or r.latex
    assert "\\beta" in S.coupling_relation().latex


def test_coupling_su2():
    r = S.coupling_relation()
    assert r.expression.rhs.subs({S.N: 2}) == 4 / S.g**2
    assert abs(r.numeric(2, 1.0) - 4.0) < 1e-12


def test_inverse_coupling_roundtrip():
    r = S.coupling_inverse()
    gv = r.numeric(2, 2.0)
    assert abs(4.0 / gv**2 - 2.0) < 1e-12


def test_trace_invariance_symbolic():
    assert S.su2_trace_conjugation_invariance() == 0


def test_su2_plaquette_term_is_one_minus_a0():
    r = S.su2_plaquette_term()
    assert sp.simplify(r.expression.rhs - (1 - sp.Symbol("a_0", real=True))) == 0


def test_ess_matches_numeric():
    r = S.ess_model()
    assert abs(r.numeric(180, 3.0) - effective_sample_size(180, 3.0)) < 1e-12


def test_efficiency_ratio_simplifies():
    r = S.efficiency_ratio()
    expected = (S.tau_M * S.t_M) / (S.tau_OR * S.t_OR)
    assert sp.simplify(r.expression.rhs - expected) == 0


def test_break_even_gives_unit_ratio():
    be = S.break_even()
    to, ttm, tto = 2.0, 1.0, 3.0
    tm = be.numeric(to, ttm, tto)
    assert abs(S.efficiency_ratio().numeric(tm, to, ttm, tto) - 1.0) < 1e-12


def test_ar1_tau_matches_numeric_estimator():
    ph = 0.6
    rho = ph ** np.arange(0, 200)
    tau = integrated_autocorrelation_time(rho, 150)
    exact = S.ar1_tau().numeric(ph)
    assert abs(float(np.ravel(tau)[0] if hasattr(tau, "__len__") else tau) - exact) < 1e-3


def test_cosh_identity_and_numeric():
    assert S.cosh_ratio_is_cosh_m()
    c = periodic_cosh_correlator(np.arange(0, 9), 1.0, 0.7, 8)
    meff = arccosh_effective_mass(c)
    assert np.allclose(meff[1:-1], 0.7)


def test_creutz_area_law_is_sigma():
    r = S.creutz_from_area_law()
    assert sp.simplify(r.expression.rhs - S.sigma) == 0
    s, pv, cv = 0.3, 0.2, 0.1
    W = lambda a, b: float(np.exp(-s * a * b - pv * (a + b) - cv))
    chi = -np.log(W(3, 3) * W(2, 2) / (W(3, 2) * W(2, 3)))
    assert abs(chi - s) < 1e-12


def test_latex_table():
    assert "\\begin{equation}" in S.latex_table()

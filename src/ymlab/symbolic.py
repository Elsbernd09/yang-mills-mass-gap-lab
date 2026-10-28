"""
Symbolic models of the equations used by the laboratory (sympy -> LaTeX).

Each model is a :class:`SymbolicResult` holding a sympy expression (or
relation), its LaTeX rendering, and a numeric callable that is checked against
the numerical ymlab implementation in the tests.  Nothing here is a claim about
the continuum Yang-Mills mass gap; these are lattice and statistics identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import sympy as sp

# --- shared symbols -------------------------------------------------------
beta, g, N = sp.symbols("beta g N", positive=True)
phi = sp.Symbol("varphi", real=True)
tau, tau_M, tau_OR = sp.symbols("tau tau_M tau_OR", positive=True)
t_M, t_OR, n_meas, t_run = sp.symbols("t_M t_OR n_meas t_run", positive=True)
sigma, R, T, p, c = sp.symbols("sigma R T p c", real=True)
m, A, Tt, t = sp.symbols("m A T_t t", positive=True)


@dataclass
class SymbolicResult:
    name: str
    expression: sp.Basic
    description: str
    numeric: Callable[..., float] | None = None
    symbols: tuple = field(default_factory=tuple)

    @property
    def latex(self) -> str:
        return sp.latex(self.expression)

    def equation_block(self) -> str:
        return "\\begin{equation}\n  " + self.latex + "\n\\end{equation}"


def coupling_relation() -> SymbolicResult:
    """beta = 2N / g^2 and its SU(2) specialisation beta = 4 / g^2."""
    expr = sp.Eq(beta, 2 * N / g**2)
    return SymbolicResult(
        "lattice coupling", expr, "Wilson coupling for SU(N)",
        lambda n, gg: 2.0 * n / gg**2, (N, g),
    )


def coupling_inverse() -> SymbolicResult:
    sol = sp.solve(sp.Eq(beta, 2 * N / g**2), g)[0]
    return SymbolicResult(
        "bare coupling from beta", sp.Eq(g, sol),
        "bare coupling solved from the Wilson coupling relation",
        lambda n, b: float(sp.sqrt(2.0 * n / b)), (N, beta),
    )


def su2_plaquette_term() -> SymbolicResult:
    """For U = a0 + i a.sigma, (1/2) Re Tr U = a0, so 1 - a0 is the cost."""
    a0, a1, a2, a3 = sp.symbols("a_0 a_1 a_2 a_3", real=True)
    U = sp.Matrix(
        [[a0 + sp.I * a3, a2 + sp.I * a1], [-a2 + sp.I * a1, a0 - sp.I * a3]]
    )
    half_re_tr = sp.simplify(sp.re(U.trace()) / 2)
    term = sp.Eq(sp.Symbol("w_p"), 1 - half_re_tr)
    det = sp.expand(U.det())
    return SymbolicResult(
        "SU(2) plaquette term", term,
        f"for a unit quaternion, $\\det U = {sp.latex(det)} = 1$",
        lambda x0: 1.0 - x0, (a0,),
    )


def su2_trace_conjugation_invariance() -> sp.Basic:
    """Return Tr(V U V^-1) - Tr(U) simplified to 0 for a generic 2x2 case."""
    u = sp.Matrix(2, 2, sp.symbols("u11 u12 u21 u22"))
    v = sp.Matrix(2, 2, sp.symbols("v11 v12 v21 v22"))
    return sp.simplify((v * u * v.inv()).trace() - u.trace())


def ess_model() -> SymbolicResult:
    """N_eff = N / (2 tau); efficiency eta = N_eff / t."""
    n_eff = n_meas / (2 * tau)
    return SymbolicResult(
        "effective sample size", sp.Eq(sp.Symbol("N_eff"), n_eff),
        "ESS from integrated autocorrelation time",
        lambda n, ta: n / (2.0 * ta), (n_meas, tau),
    )


def efficiency_ratio() -> SymbolicResult:
    """R = eta_OR / eta_M reduces to (tau_M t_M)/(tau_OR t_OR)."""
    eta_m = n_meas / (2 * tau_M) / t_M
    eta_or = n_meas / (2 * tau_OR) / t_OR
    ratio = sp.simplify(eta_or / eta_m)
    return SymbolicResult(
        "efficiency ratio", sp.Eq(sp.Symbol("R"), ratio),
        "paired ESS-per-second ratio; independent of the number of measurements",
        lambda tm, to, ttm, tto: (tm * ttm) / (to * tto),
        (tau_M, tau_OR, t_M, t_OR),
    )


def break_even() -> SymbolicResult:
    """R > 1  <=>  tau_M / tau_OR > t_OR / t_M (gain exceeds cost penalty)."""
    sol = sp.solve(sp.Eq((tau_M * t_M) / (tau_OR * t_OR), 1), tau_M)[0]
    return SymbolicResult(
        "break-even", sp.Eq(tau_M, sol),
        "Metropolis autocorrelation time at which the efficiency ratio equals one",
        lambda to, ttm, tto: to * tto / ttm, (tau_OR, t_M, t_OR),
    )


def ar1_tau() -> SymbolicResult:
    """Integrated autocorrelation time of an AR(1) chain with rho_k = phi^k."""
    k = sp.Symbol("k", integer=True, nonnegative=True)
    total = sp.Rational(1, 2) + sp.summation(phi**(k + 1), (k, 0, sp.oo))
    closed = sp.simplify(
        sp.Rational(1, 2) + phi / (1 - phi)
    )
    target = (1 + phi) / (2 * (1 - phi))
    assert sp.simplify(closed - target) == 0
    del total
    return SymbolicResult(
        "AR(1) integrated autocorrelation time", sp.Eq(sp.Symbol("tau_int"), target),
        "geometric sum of autocorrelations for $|\\varphi|<1$",
        lambda ph: (1.0 + ph) / (2.0 * (1.0 - ph)), (phi,),
    )


def cosh_effective_mass() -> SymbolicResult:
    """[C(t-1)+C(t+1)] / (2 C(t)) = cosh(m) for C = A(e^{-mt}+e^{-m(T-t)})."""
    C = lambda s: A * (sp.exp(-m * s) + sp.exp(-m * (Tt - s)))
    ratio = sp.simplify(
        ((C(t - 1) + C(t + 1)) / (2 * C(t))).rewrite(sp.cosh)
    )
    ratio = sp.simplify(ratio)
    return SymbolicResult(
        "cosh effective mass", sp.Eq(sp.Symbol("m_eff"), sp.acosh(sp.cosh(m))),
        f"the estimator ratio $[C(t-1)+C(t+1)]/2C(t)$ equals ${sp.latex(ratio)}$, independent of $t$",
        lambda mm: float(sp.acosh(sp.cosh(mm))), (m,),
    )


def cosh_ratio_is_cosh_m() -> bool:
    C = lambda s: A * (sp.exp(-m * s) + sp.exp(-m * (Tt - s)))
    ratio = (C(t - 1) + C(t + 1)) / (2 * C(t))
    # Substitute numeric points; the identity holds for all t, T_t.
    for tv, Tv in ((3, 8), (2, 6), (5, 12)):
        val = ratio.subs({t: tv, Tt: Tv, A: 1, m: sp.Rational(7, 10)})
        if abs(float(val) - float(sp.cosh(sp.Rational(7, 10)))) > 1e-12:
            return False
    return True


def creutz_from_area_law() -> SymbolicResult:
    """chi(R,T) of W = exp(-sigma R T - p(R+T) - c) is exactly sigma."""
    def W(r, tt):
        return sp.exp(-sigma * r * tt - p * (r + tt) - c)

    chi = -sp.log(
        W(R, T) * W(R - 1, T - 1) / (W(R, T - 1) * W(R - 1, T))
    )
    chi = sp.simplify(sp.expand_log(chi, force=True))
    return SymbolicResult(
        "Creutz area law", sp.Eq(sp.Symbol("chi"), chi),
        "perimeter and constant terms cancel in the Creutz ratio",
        lambda s: s, (sigma,),
    )


MODELS = (
    coupling_relation, coupling_inverse, su2_plaquette_term, ess_model,
    efficiency_ratio, break_even, ar1_tau, cosh_effective_mass,
    creutz_from_area_law,
)


def all_models() -> list[SymbolicResult]:
    return [f() for f in MODELS]


def latex_table(results: list[SymbolicResult] | None = None) -> str:
    """Render the models as a LaTeX ``description``-style list of equations."""
    results = results if results is not None else all_models()
    lines = []
    for r in results:
        lines.append(f"\\paragraph{{{r.name[0].upper() + r.name[1:]}.}} {r.description}.")
        lines.append(r.equation_block())
    return "\n".join(lines)

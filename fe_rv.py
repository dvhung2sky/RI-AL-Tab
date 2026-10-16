"""Shared random-variable model for the FE reliability benchmarks (multi-RV redesign).

Every truss input vector U (standard normal columns) is laid out as
    U = [ load RVs ... , E , Fy , A_group_1 , ... , A_group_G ]
Loads keep each module's own normal model (loads_from_U). This module maps the
material + section columns to physical multiplicative factors:
  - E  : global Young's modulus     ~ Lognormal(mean 1, CoV 0.05)
  - Fy : global yield strength      ~ Lognormal(mean 1, CoV 0.08)
  - A  : per-GROUP section area      ~ Lognormal(mean 1, CoV 0.05), scaled by a DISTINCT
         nominal area per group (elements in different groups have different sections).
Standard JCSS-style CoVs. Positive quantities are lognormal so they never go <= 0.
"""
import numpy as np

COV_E, COV_FY, COV_A = 0.05, 0.08, 0.05


def _logn(u, cov):
    """Standard-normal column -> mean-1 lognormal multiplicative factor at given CoV."""
    s = np.sqrt(np.log(1.0 + cov ** 2))
    return np.exp(s * u - 0.5 * s ** 2)


def n_rv(n_load, n_groups):
    """Total input dimension = load RVs + E + Fy + area groups."""
    return n_load + 2 + n_groups


def material_section(U, n_load, nominal_A, group_of_member):
    """Map U's material/section columns to physical per-sample properties.
    U        : (n, n_rv) standard normal; columns [loads..., E, Fy, A_g1..A_gG]
    nominal_A: (G,) DISTINCT nominal group areas
    group_of_member: (M,) int, area-group index of each member
    Returns  E:(n,), Fy:(n,), A_member:(n, M)  (all physical, mean = nominal).
    """
    nominal_A = np.asarray(nominal_A, float)
    group_of_member = np.asarray(group_of_member, int)
    G = len(nominal_A)
    E = _logn(U[:, n_load], COV_E)
    Fy = _logn(U[:, n_load + 1], COV_FY)
    A_grp = nominal_A[None, :] * _logn(U[:, n_load + 2:n_load + 2 + G], COV_A)   # (n, G)
    A_member = A_grp[:, group_of_member]                                        # (n, M)
    return E, Fy, A_member


def nominal_material_section(nominal_A, group_of_member):
    """Deterministic nominal props (E=Fy=factor 1) for the calibration linear bracket."""
    nominal_A = np.asarray(nominal_A, float)
    group_of_member = np.asarray(group_of_member, int)
    return 1.0, 1.0, nominal_A[group_of_member]      # E_factor, Fy_factor, A_member (nominal)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    U = rng.standard_normal((100000, 2 + 2 + 3))
    E, Fy, A = material_section(U, 2, [1.4, 1.0, 0.7], [0, 0, 1, 2, 2])
    assert abs(E.mean() - 1) < 0.01 and abs(Fy.mean() - 1) < 0.01, (E.mean(), Fy.mean())
    assert (E > 0).all() and (Fy > 0).all() and (A > 0).all()
    # per-group means match distinct nominals; members share their group's column
    assert abs(A[:, 0].mean() - 1.4) < 0.02 and abs(A[:, 3].mean() - 0.7) < 0.02
    assert np.array_equal(A[:, 3], A[:, 4])          # same group -> same area draw
    assert abs(np.std(np.log(A[:, 0])) - np.sqrt(np.log(1 + 0.05 ** 2))) < 0.01
    print("fe_rv OK: mean-1 lognormal factors, distinct group nominals, positivity")

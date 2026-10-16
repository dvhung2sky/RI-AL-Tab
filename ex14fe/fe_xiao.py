"""Ex4 reliability benchmark: Xiao et al. (2015) half-scale 3-storey RC building, side column D3 removed.

Deterministic core = `xiao2015_building.py` (read-only here), reference R2 of CALIBRATION_BUILDING.md.
Intent: A11/Submission/intent/xiao-building-case-study.md.

Physics per sample: D3 is removed BEFORE any load; the whole Test-2 gravity (self-weight + 20 kN/m2 on
the T area, all three floors) is ONE pattern ramped from zero under displacement control of the D3
joint (2nd floor). LF = 1 is the test state, and the curve starts at zero drop and zero load, which the
energy balance P_ps(u) = (1/u) int_0^u R du needs. The FE side stores the RAW curve (u, LF); P_ps,
u_fr and the margins are post-processing, so every demand-side choice stays free after generation.

RANDOM VARIABLES (13: 9 structural -> OpenSees, 4 demand -> post-processing). Lognormal, `_ln(u, cov,
mean)`: the R2 value is the MEAN (median = mean exp(-s^2/2)), as in Ex2/Ex3. CANDIDATES for the screen.
    U0 f_c      x FCU (all storeys)   0.15      U5 f_t mult   FT_MULT 1.0   0.20
    U1 f_y      x FY  (all bars)      0.06      U6 GJ_FAC     0.5 [ASSUME]  0.30
    U2 f_u/f_y  yield-ratio mult      0.05      U7 COVER      20 mm         0.15
    U3 eps_su   ESU 0.10 [ASSUME]     0.20      U8 E_s        200 GPa       0.03
    U4 E_c mult EC_MULT 1.0           0.10      U9-U12  theta_R, D, L, theta_allow (margins only)
Not an RV: slab thickness (it is in the self-weight, so it would move the load pattern W, coupling a
structural RV to the demand); the [CAL] slab tension-stiffening knee SLAB_TS (the record finds the
response not monotone-smooth in it).

    python3 fe_xiao.py            # one run at the R2 defaults, timed
"""
import os
import sys
import time
import numpy as np
import openseespy.opensees as ops
import xiao2015_building as xb

U_MAX = float(os.environ.get("XUMAX", "450.0"))  # ponytail: window = the existing pushdown's u_stop; the pilot decides
DU = float(os.environ.get("XDU", "4.0"))         # MEASURED (nominal, same load): du 4 + tol 1e-3 is 2.3x faster than
                                                # du 2 + tol 1e-4 (100 vs 236 s); P_ps max -0.013 %, P_ps on the 2 mm grid
                                                # median 0.019 %, d* identical at 5 demand levels. SuperLU/SparseSYM fail,
                                                # BandGen 2x slower, Krylov -maxDim 6 and initial-tangent iteration slower
T_MAX = float(os.environ.get("XTMAX", "1800"))   # s per sample; a stuck sample ends as "timeout" and is dropped
TOL = float(os.environ.get("XTOL", "1e-3"))    # NormDispIncr, mm (xb.pushdown() uses 1e-4; see DU)
ALG = os.environ.get("XALG", "KrylovNewton")    # MEASURED (nominal, 2026-09-19): Newton died at 138 mm after
                                                # 1162 s; KrylovNewton reached 450 mm in 118 s, curves within 0.19 %
if os.environ.get("XIMPLEX", "0") == "0":         # MEASURED (vast pilot, 2026-09-19): with R2's IMPL-EX slab 64 of 191
    xb.SLAB_OPTS = ()                            # runs died at 81-184 mm, on the load plateau; implicit ASDConcrete3D
                                                 # carries 4 of 4 of them to 450 mm with no rescues, P_ps max -0.18 %
NRV, N_STRUCT = 13, 9
NAMES = ["f_c", "f_y", "fu/fy", "eps_su", "E_c", "f_t", "GJ", "cover", "E_s", "theta_R", "D", "L", "theta_allow"]
_COV = dict(fc=0.15, fy=0.06, ratio=0.05, esu=0.20, ec=0.10, ft=0.20, gj=0.30, cov=0.15, Es=0.03,
            thR=0.18, D=0.10, L=0.15, thA=0.30)
_BASE = dict(FCU=dict(xb.FCU), FY=dict(xb.FY), FU=dict(xb.FU), ESU=xb.ESU, EC_MULT=xb.EC_MULT,
             FT_MULT=xb.FT_MULT, GJ_FAC=xb.GJ_FAC, COVER=xb.COVER, ES=xb.ES)


def _ln(u, cov, mean=1.0):
    s = np.sqrt(np.log(1.0 + cov ** 2))
    return mean * np.exp(s * u - 0.5 * s ** 2)


def set_rvs(u_row):
    """Structural RVs -> module globals of xiao2015_building, ALWAYS rebuilt from the R2 base, so a
    worker that runs many samples carries nothing from one to the next."""
    B = _BASE
    m = _ln(u_row[0], _COV["fc"]); xb.FCU = {k: v * m for k, v in B["FCU"].items()}
    m = _ln(u_row[1], _COV["fy"]); xb.FY = {d: v * m for d, v in B["FY"].items()}
    r = _ln(u_row[2], _COV["ratio"]); xb.FU = {d: xb.FY[d] * B["FU"][d] / B["FY"][d] * r for d in B["FU"]}
    xb.ESU = _ln(u_row[3], _COV["esu"], B["ESU"])
    xb.EC_MULT = _ln(u_row[4], _COV["ec"], B["EC_MULT"])
    xb.FT_MULT = _ln(u_row[5], _COV["ft"], B["FT_MULT"])
    xb.GJ_FAC = _ln(u_row[6], _COV["gj"], B["GJ_FAC"])
    xb.COVER = _ln(u_row[7], _COV["cov"], B["COVER"])
    xb.ES = _ln(u_row[8], _COV["Es"], B["ES"])


def pushdown(u_max=U_MAX, du=DU, verbose=False):
    """Returns (u [mm], LF, W [N] = pattern total at LF 1, reason in {window, collapse, numerical, timeout})."""
    xb.build(); xb.add_test_load("test2")
    for e in xb.COL_ELE["D3"]:
        ops.remove("element", e)
    for n in xb.COL_MID["D3"]:
        ops.remove("node", n)
    ops.timeSeries("Linear", 1); ops.pattern("Plain", 1, 1)
    for n, w in xb.LOADED.items():
        ops.load(n, 0.0, 0.0, -w, 0.0, 0.0, 0.0)
    ctrl = xb.MON["D3"]
    ops.system("UmfPack"); ops.numberer("RCM"); ops.constraints("Plain")
    ops.test("NormDispIncr", TOL, 50, 0); ops.algorithm(ALG)
    ops.integrator("DisplacementControl", ctrl, 3, -du); ops.analysis("Static")
    u, lf, reason, t0 = [0.0], [0.0], "window", time.time()
    while u[-1] < u_max - 1e-6:
        ok = ops.analyze(1)
        if ok != 0:                                  # f < 1 is a part step
            # MEASURED: ModifiedNewton never rescued a step here and returned NaN; Krylov sub-steps do
            for alg, f in (("KrylovNewton", 0.5), ("KrylovNewton", 0.25), ("NewtonLineSearch", 0.25),
                           ("KrylovNewton", 0.1), ("KrylovNewton", 0.02)):
                ops.algorithm(alg); ops.test("NormDispIncr", 1.0e-3, 100, 0)
                ops.integrator("DisplacementControl", ctrl, 3, -du * f)
                if (ok := ops.analyze(1)) == 0:
                    break
            ops.algorithm(ALG); ops.test("NormDispIncr", TOL, 50, 0)
            ops.integrator("DisplacementControl", ctrl, 3, -du)
            if ok != 0:
                reason = "numerical"; break
        u.append(-ops.nodeDisp(ctrl, 3)); lf.append(ops.getLoadFactor(1))
        if verbose and len(u) % 10 == 0:
            print(f"  u {u[-1]:6.1f} mm  LF {lf[-1]:.3f}  [{time.time() - t0:.0f} s]", flush=True)
        if lf[-1] < 0.3 * max(lf):
            reason = "collapse"; break
        if time.time() - t0 > T_MAX:
            reason = "timeout"; break
    return np.array(u), np.array(lf), float(sum(xb.LOADED.values())), reason


def response(u_row):
    """One sample: raw curve + termination reason. Post-processing happens later, on the stored curves."""
    set_rvs(u_row)
    t = time.time()
    u, lf, W, why = pushdown()
    return dict(u=u.astype(np.float32), lf=lf.astype(np.float32), W=W, reason=why, wall=time.time() - t)


# ------------------------------------------------------------------------------------------------
# DEMAND SIDE -- free after generation (build_xiao_pop.py assembles it). Driver contract, same names
# and same margin form as fe_rbs.py, so ex3_phys.py / ak_mcs.py / phys_mine_replay.py port unchanged.
GRID = np.arange(0.0, 448.0 + 1e-9, 4.0)           # inside every window run (min u_last 450.0 mm)
CACHE = {}
L_CLEAR = 4570.0                                   # [CHOICE, intent Open question 5] the D-line bay
BASE = {"Lc": L_CLEAR}                             # D2-D3-D4: the double-span bridging mechanism over
                                                   # the removed column, so its chord rotation is the
                                                   # acceptance check. MEASURED alternative: the 3050 mm
                                                   # transverse bay C3-D3 rotates 1.5x more at the same
                                                   # drop and makes acceptance govern 96 of 99 failures,
                                                   # i.e. a one-mode population. 4570 gives 28 / 72.
_THA_MED = float(os.environ.get("THA_MED", "0.085"))   # Ex2's RC acceptance rotation, a MEAN not a median
_WD, _WL = 0.60, 0.40                              # as Ex2/Ex3
_UNOM = 300.0                                      # as Ex3. The failure SET {min(g) <= 0} is invariant
                                                   # to it; it only scales which mode is reported away
                                                   # from the boundary.
TARGET_PF = float(os.environ.get("TARGET_PF", "1e-2"))


def curve(u, lf, W):
    """Stored raw curve -> (P_ps on GRID, max P_ps, u_fr, u_last). R(u) = lf(u) * W.

    u_fr is inf for every Ex4 run: displacement control traced the descending branch to 450 mm and the
    worst single-step load drop in the population is 3.9 %, so no bar-fracture truncation occurs and
    mode 2 cannot govern. Ex4 is a TWO-mode population (strength, acceptance) -- reported, not forced
    to three (intent, `Limit state`). The row is kept so the driver readouts index modes 1..3 unchanged.
    """
    u = np.asarray(u, float); R = np.asarray(lf, float) * W
    Pd = np.array([np.trapezoid(R[:i + 1], u[:i + 1]) / x if x > 1e-9 else 0.0 for i, x in enumerate(u)])
    return np.interp(GRID, u, Pd).astype(np.float32), float(Pd.max()), float("inf"), float(u[-1])


def margins(U, Pdg, Pdmax, ufr, ulast, lam):
    """Ex2/Ex3 form exactly: g = min(strength, rupture, acceptance), normalised."""
    thR = _ln(U[:, 9], _COV["thR"])
    dem = lam * (_WD * _ln(U[:, 10], _COV["D"]) + _WL * _ln(U[:, 11], _COV["L"]))
    thA = _ln(U[:, 12], _COV["thA"], _THA_MED)
    g1 = (thR * Pdmax - dem) / lam
    hit = (thR[:, None] * Pdg) >= dem[:, None]
    dstar = np.where(hit.any(1), GRID[hit.argmax(1)], ulast)
    G = np.vstack([g1, (ufr - dstar) / _UNOM, (thA * L_CLEAR - dstar) / _UNOM])
    return G.min(0), (G.argmin(0) + 1).astype(np.int8)


def make_world(seed, noise):
    d = CACHE[seed]
    g = d["m"] / d["S0"]
    return d["U"], g.copy(), g, float(np.mean(g <= 0))


if __name__ == "__main__":
    t0 = time.time()
    u, lf, W, why = pushdown(verbose=True)
    wall = time.time() - t0
    Pd = np.array([np.trapezoid(lf[:i + 1], u[:i + 1]) / x if x > 1e-9 else 0.0 for i, x in enumerate(u)])
    i1 = int(np.argmax(Pd >= 1.0)) if (Pd >= 1.0).any() else -1
    print(f"nominal: {why} at u {u[-1]:.1f} mm, {len(u) - 1} steps, {wall:.0f} s ({wall / max(len(u) - 1, 1):.2f} s/step) | "
          f"W {W / 1e3:.0f} kN | LF peak {lf.max():.3f} at {u[lf.argmax()]:.0f} mm | P_ps max {Pd.max():.3f} | "
          f"P_ps = 1 at u {u[i1] if i1 >= 0 else float('nan'):.1f} mm (R2 transient D3 peak 27.2)")
    np.savez(sys.argv[1] if len(sys.argv) > 1 else "fe_xiao_nominal.npz", u=u, lf=lf, W=W, wall=wall)

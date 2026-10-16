"""Ex14-FE-RBS reliability benchmark: steel RBS beam-column assembly under column removal.

Deterministic core = `nist_assembly.py` (read-only here), calibrated against the RBS specimen of
NIST TN 1669 (Sadek, Lew, Main, Robert & Chiarito 2010) -- two 20 ft spans, centre stub column
pushed down under displacement control. Status: CALIBRATED, 8 of 8 scored quantities inside the
pre-stated band for both specimens (`CALIBRATION.md`).

Same limit state as Ex14-FE-Yi2008, so the three examples are directly comparable:

    fail  <=>  r > C(x_s, theta_allow),   r = lam (0.6 D + 0.4 L) / theta_R
    C = min(C_fr, C_acc),  C_fr = max_{d < u_fr} P_ps(d),  C_acc = max_{d < theta_allow*l_n} P_ps(d)
    P_ps(d) = (1/d) integral_0^d R(u) du            energy balance, Izzuddin et al. (2008)

RANDOM VARIABLES (13: 9 structural -> OpenSees, 4 demand -> integrated exactly)
    U0 f_y       LN 0.06   beam flange yield, 413.5 MPa   [Table 3-1, scaled coupon]
    U1 f_u/f_y   LN 0.05   yield RATIO, 1.329             see NOTE 1
    U2 eps_su    LN 0.15   uniform elongation, 0.1317
    U3 f_y,col   LN 0.06   column yield, 378 MPa          (panel zone only)
    U4 E_s       LN 0.03   200 GPa                        see NOTE 2
    U5 theta_u   LN 0.30   fracture rotation, 0.1615 rad  [CAL]  -- dominant, 26 % swing
    U6 K_br      LN 0.40   end restraint, 35 kN/mm        [ASSUME]
    U7 t_doubler LN 0.05   doubler plate, 14.29 mm        [READ, Fig. 3-2]   see NOTE 2
    U8 c_fac     LN 0.05   RBS cut per side, 0.25 b_f     [AISC 358 tolerance]
    U9  theta_R   LN 0.18   model uncertainty
    U10 D         LN 0.10   dead load
    U11 L         LN 0.15   live load
    U12 theta_all LN 0.30   acceptance rotation, mean 0.127 rad   see NOTE 3

NOTE 1  f_u is the yield RATIO times f_y, not an independent variable. As two independent
    lognormals P(f_u < f_y) = 1.34e-4, i.e. about 7 of 50,000 samples with an ultimate strength
    below yield, which is not a steel. Both come from one coupon. Ex2 used a clamp
    (FU = max(..., 1.02*FY)); the ratio is cleaner because it piles no mass on the clamp.

NOTE 2  TWO of the nine structural RVs are MEASURED INERT: E_s (0.6 % swing at +/-1.5 sigma) and
    t_doubler (0.04 %, because the doubler-plated panel zone never yields). Every other structural RV
    moves Pd_max by 2-26 %. Both are kept: they are real physical variables, Ex2 carries E_s at the
    same CoV, and known-irrelevant inputs are a fair test of a surrogate's ability to ignore noise
    dimensions. They are declared, not hidden.

    L_conn (the [ASSUME] fracture gauge length, 0.5 d_b) was TRIED in U7's place at CoV 0.20 and
    REJECTED after 5,000 production samples. Two independent reasons, the second decisive:
      (a) Geometric. build() places the gauge inside the RBS cut as `t = 0.5*(b_fac*d - L_conn)`,
          a sliver each side. Above L_conn = b_fac*d = 432.0 mm the slivers invert and the mesh is
          invalid -- "the Algorithm failed at step: 0". That is z = 1.798, and P(z > 1.798) = 3.61 %
          against an observed 3.80 % drop, ALL of it in L_conn's upper tail (mean z of the dropped
          samples +2.15, none below +1.5). That is not thinning, it is TRUNCATION: the surviving
          population is no longer the one the seed defines. A clamp would have fixed this.
      (b) Calibration. theta_u was calibrated AT L_conn = 0.5 d_b, and TN 1669 Sec. 5.2.1 states
          eps_f is a mesh-dependent quantity. Varying the gauge length therefore silently
          invalidates the calibration that licenses this model. No clamp fixes that.
    Net effect of reverting: the population loses a live RV (5.7 % / 5.5 % swing) and keeps an inert
    one. That is the honest trade, and it is recorded rather than quietly repaired.

NOTE 3  The acceptance rotation for COLUMN REMOVAL is disputed across code generations by about a
    factor of three: GSA 2003 allowed a blanket 12 deg = 0.209 rad plastic rotation in steel beams,
    while UFC 4-023-03 / ASCE 41 replaced that with component criteria of order 0.06-0.09 rad for
    RBS connections -- and TN 1669 is one of the tests that drove the change, the WUF-B specimen
    having fractured at 0.081 rad, far short of the blanket limit. So theta_allow is not a number to
    look up; it is uncertain. Mean 0.127 rad with CoV 0.30 puts p5 = 0.075 (the ASCE 41 criterion),
    median = 0.122 (the measured median fracture rotation) and p95 = 0.197 (just under GSA's).
    _THA_MED is free to re-sweep: it never reaches OpenSees, only `margins`.

TWO NAMES ARE DELIBERATELY MISLEADING, so the drivers work unmodified:
    L_CLEAR = 6096.0 is the CENTRE-TO-CENTRE span, not a clear span. It is NIST's own chord length:
             495/6096 = 0.0812 and 851/6096 = 0.1396 reproduce the reported 0.081 and 0.140 rad.
    _THA_MED is a lognormal MEAN, not a median (`_ln` returns mean*exp(s*u - 0.5 s^2)).

Env: NPOP, PF, JOBS, THA_MED, UMAX, DU.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import multiprocessing as mp
import nist_assembly as A

NRV, N_STRUCT = 13, 9
TARGET_PF = float(os.environ.get("PF", "3e-3"))
N_POP = int(float(os.environ.get("NPOP", "10000")))
JOBS = int(os.environ.get("JOBS", "8"))
# Analysis window. NOT set by numerical reach, as Ex2's was -- MEASURED: 0 divergences in 1,200
# samples at both 1000 and 1300 mm. It is set by the READOUT: the structure-exploiting readout fits
# its PCA of log Pd on NON-ruptured runs (`tr_ = ev[~isfinite(UFR[ev])]`), and at 1300 mm only 4.4 %
# of runs survive un-fractured -- 13 training curves at 300 FE, which starves it. At 1000 mm it is
# 25 %, about 75 curves. 1000 mm is also the value already in nist_assembly.RBS, set during
# calibration, so it is not tuned for the benchmark.
U_MAX = float(os.environ.get("UMAX", "1000.0"))
DU = float(os.environ.get("DU", "1.0"))
L_CLEAR = 6096.0                                   # centre-to-centre; see the docstring warning
BASE = {"Lc": L_CLEAR}                             # driver compatibility: reads Mod.BASE["Lc"]
# A fracture call below this is a numerical wobble, not a rupture. Ex2's 30 mm guard on a 2467 mm
# span, scaled to 6096 mm. MEASURED INACTIVE: the pilot p5 of u_fr is 481 mm, so it never fires --
# unlike Ex2, where 96 of 19,803 fired trivially early. Kept as a tripwire, not as a tuned threshold.
U_FR_MIN = 100.0
GRID = np.arange(0.0, U_MAX + 1e-9, 2.0)
CACHE = {}
NAMES = ["f_y", "f_u/f_y", "eps_su", "f_y_col", "E_s", "theta_u", "K_br", "t_doubler", "c_fac",
         "theta_R", "D", "L", "theta_allow"]
_COV = dict(fy=0.06, ryr=0.05, esu=0.15, fyc=0.06, Es=0.03, thu=0.30, Kbr=0.40, tdb=0.05,
            cfac=0.05, thR=0.18, D=0.10, L=0.15, thA=0.30)
_THA_MED = float(os.environ.get("THA_MED", "0.127"))     # a MEAN; see the docstring warning
_WD, _WL = 0.60, 0.40
_UNOM = 300.0
_ES0 = 200000.0
_RYR = A.RBS["fu"] / A.RBS["fy"]                   # 1.329, the calibrated yield ratio
TAG = "" if abs(_THA_MED - 0.127) < 1e-12 else f"_t{_THA_MED:g}"


def _ln(u, cov, mean=1.0):
    s = np.sqrt(np.log(1.0 + cov ** 2))
    return mean * np.exp(s * u - 0.5 * s ** 2)


def response(u_row):
    """One RBS pushdown. Returns (P_ps on GRID, max P_ps, u_fr, u_last) or None."""
    p = dict(A.RBS)
    p["fy"] = A.RBS["fy"] * _ln(u_row[0], _COV["fy"])
    p["fu"] = p["fy"] * _RYR * _ln(u_row[1], _COV["ryr"])        # ratio, so f_u > f_y always
    p["esu"] = A.RBS["esu"] * _ln(u_row[2], _COV["esu"])
    p["fy_col"] = A.RBS["fy_col"] * _ln(u_row[3], _COV["fyc"])
    p["theta_u"] = A.RBS["theta_u"] * _ln(u_row[5], _COV["thu"])
    p["K_br"] = A.RBS["K_br"] * _ln(u_row[6], _COV["Kbr"])
    p["t_doubler"] = A.RBS["t_doubler"] * _ln(u_row[7], _COV["tdb"])
    p["c_fac"] = A.RBS["c_fac"] * _ln(u_row[8], _COV["cfac"])
    p["u_max"], p["du"] = U_MAX, DU
    # module global, read inside build(); safe under both fork and spawn because the assignment
    # happens inside the worker, per call. Verified: K0 scales 0.510 / 1.000 / 1.970 with ES.
    A.ES = _ES0 * _ln(u_row[4], _COV["Es"])
    try:
        u, R, N, dx, th, ri = A.pushdown(p)
    except Exception:
        return None
    if len(u) < 20:
        return None
    # WHY THE RUN STOPPED. pushdown() breaks on four different conditions and returns no reason, so
    # this is re-derived from its own break conditions, in their order. Getting this wrong is the
    # single most dangerous failure mode in this file: Ex2's rule (u[-1] < 0.99*U_MAX -> None) would
    # discard every FRACTURED run, which is 75 % of this population.
    if th[-1] >= p["theta_u"]:                 why = 1       # rotation criterion = connection fracture
    elif len(R) > 20 and R[-1] < 0.3 * R.max(): why = 2      # resistance collapse
    elif u[-1] >= U_MAX - 2.0 * DU:            why = 3       # reached the window intact
    else:                                      why = 4       # divergence / equilibrium check
    if why == 4:
        return None
    Pd = np.array([np.trapezoid(R[:i + 1], u[:i + 1]) / x if x > 1e-9 else 0.0
                   for i, x in enumerate(u)])
    Pdg = np.interp(GRID, u, Pd)
    tail = GRID > u[-1]
    if tail.any() and why in (1, 2):
        # after fracture R ~ 0, so Pd(u) = (1/u) int_0^u R du decays as Pd(u_fr)*u_fr/u. Ex2 holds
        # Pd[-1] flat (`right=Pd[-1]`), which is harmless there because its runs always reach the
        # window, but here it would assert the assembly keeps its capacity after the connection has
        # fractured. That is wrong and unconservative, and it changes d* for exactly the samples
        # whose first capacity/demand crossing falls after fracture -- the rupture failures.
        Pdg[tail] = Pd[-1] * u[-1] / GRID[tail]
    # u_fr = inf when the assembly did NOT rupture inside the window, so mode 2 simply never governs
    # -- never u_last, which is a numerical quantity. Every rupture miner in the drivers keys off
    # ~isfinite(UFR); nist_assembly.metrics() must NOT be used here, its u_fr falls back to u[-1].
    u_fr = float(u[-1]) if (why in (1, 2) and u[-1] >= U_FR_MIN) else float("inf")
    return Pdg.astype(np.float32), float(Pd.max()), u_fr, float(u[-1])


def margins(U, Pdg, Pdmax, ufr, ulast, lam):
    thR = _ln(U[:, 9], _COV["thR"])
    dem = lam * (_WD * _ln(U[:, 10], _COV["D"]) + _WL * _ln(U[:, 11], _COV["L"]))
    thA = _ln(U[:, 12], _COV["thA"], _THA_MED)
    g1 = (thR * Pdmax - dem) / lam
    hit = (thR[:, None] * Pdg) >= dem[:, None]
    dstar = np.where(hit.any(1), GRID[hit.argmax(1)], ulast)
    G = np.vstack([g1, (ufr - dstar) / _UNOM, (thA * L_CLEAR - dstar) / _UNOM])
    return G.min(0), (G.argmin(0) + 1).astype(np.int8)


def _chunk(args):
    lo, Us = args
    return lo, [response(u) for u in Us]


def make_world(seed, noise):
    d = CACHE[seed]
    g = d["m"] / d["S0"]
    return d["U"], g.copy(), g, float(np.mean(g <= 0))


def demo():
    """Self-check: the four things that would silently corrupt the population if they broke."""
    r = response(np.zeros(NRV))
    assert r is not None and abs(r[1] / 1e3 - 1301) < 15, f"u=0 Pd_max {r[1]/1e3:.0f}, expected ~1301"
    assert abs(r[2] - 817) < 8, f"u=0 u_fr {r[2]:.0f}, expected ~817"
    assert len(r[0]) == len(GRID) == 501
    hi = np.zeros(NRV); hi[5] = 3.0                      # theta_u at +3 sigma: cannot fracture in 1000 mm
    rh = response(hi)
    assert rh is not None and not np.isfinite(rh[2]), f"+3sig theta_u gave finite u_fr {rh[2]}"
    assert rh[3] >= U_MAX - 2 * DU, "a non-fracturing run must reach the window"
    lo = np.zeros(NRV); lo[5] = -2.0                     # early fracture: Pd tail must DECAY, not hold
    rl = response(lo)
    assert np.isfinite(rl[2]) and rl[2] < 600, f"-2sig theta_u u_fr {rl[2]:.0f}, expected early"
    t = GRID > rl[3]
    assert t.any() and np.all(np.diff(rl[0][t]) < 0), "Pd past fracture must decay, not hold flat"
    U = np.random.default_rng(1).standard_normal((20000, NRV))
    fy = A.RBS["fy"] * _ln(U[:, 0], _COV["fy"])
    fu = fy * _RYR * _ln(U[:, 1], _COV["ryr"])
    assert (fu > fy).all(), f"{int((fu <= fy).sum())} draws with f_u <= f_y"
    print(f"demo OK  u=0: Pd_max {r[1]/1e3:.0f} kN  u_fr {r[2]:.0f} mm | "
          f"+3sig: u_fr inf, u_last {rh[3]:.0f} | -2sig: u_fr {rl[2]:.0f}, tail decays | f_u > f_y 20k/20k")


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "demo":
        demo(); sys.exit()
    if sys.argv[1:] and sys.argv[1] == "probe":
        Us = np.random.default_rng(0).standard_normal((3, NRV))
        for k in range(3):
            t0 = time.time(); r = response(Us[k]); dt = time.time() - t0
            print(f"  sample {k}: {dt:5.2f} s  " + ("FAILED" if r is None else
                  f"P_ps,max={r[1]/1e3:6.1f} kN  u_fr={r[2]:5.0f}  u_last={r[3]:5.0f} mm"), flush=True)
        sys.exit()
    t0 = time.time()
    U = np.random.default_rng(0).standard_normal((N_POP, NRV))
    rfn = f"resp_rbs_N{N_POP}.npy"
    if os.path.exists(rfn):
        r = np.load(rfn, allow_pickle=True).item()
        Pdg, Pdmax, ufr, ulast, bad = r["Pd"], r["Pdmax"], r["ufr"], r["ulast"], r["bad"]
    else:
        print(f"evaluating {N_POP:,} RBS pushdowns on {JOBS} workers "
              f"(u_max={U_MAX}, du={DU}) ...", flush=True)
        step = max(1, N_POP // 200)
        tasks = [(lo, U[lo:lo + step]) for lo in range(0, N_POP, step)]
        Pdg = np.zeros((N_POP, len(GRID)), np.float32)
        Pdmax = np.zeros(N_POP); ufr = np.zeros(N_POP); ulast = np.zeros(N_POP)
        bad = np.zeros(N_POP, bool); done = 0
        seen = np.zeros(N_POP, bool)
        pfn = rfn.replace(".npy", "_partial.npy")
        if os.path.exists(pfn):                       # resume: never lose a stopped run
            q = np.load(pfn, allow_pickle=True).item()
            Pdg, Pdmax, ufr, ulast, bad, seen = q["Pd"], q["Pdmax"], q["ufr"], q["ulast"], q["bad"], q["seen"]
            tasks = [t for t in tasks if not seen[t[0]]]
            done = int(seen.sum()); print(f"  resuming: {done:,} already done", flush=True)
        with mp.Pool(JOBS) as pool:
            for lo, out in pool.imap_unordered(_chunk, tasks):
                for k, rr in enumerate(out):
                    i = lo + k
                    if rr is None: bad[i] = True; continue
                    Pdg[i], Pdmax[i], ufr[i], ulast[i] = rr
                for k in range(len(out)): seen[lo + k] = True
                done += len(out)
                if done % 5000 < len(out):
                    np.save(pfn, dict(U=U, grid=GRID, Pd=Pdg, Pdmax=Pdmax, ufr=ufr, ulast=ulast,
                                      bad=bad, seen=seen), allow_pickle=True)
                el = time.time() - t0
                print(f"  {done:,}/{N_POP:,} ({int(bad.sum())} failed) [{el/60:.1f} min, "
                      f"eta {el/max(done,1)*(N_POP-done)/60:.0f} min]", flush=True)
        np.save(rfn, dict(U=U, grid=GRID, Pd=Pdg, Pdmax=Pdmax, ufr=ufr, ulast=ulast, bad=bad),
                allow_pickle=True)
    ok = ~bad
    print(f"  excluding {int(bad.sum()):,} non-convergent samples ({bad.mean()*100:.2f} %); "
          f"population renormalised to {int(ok.sum()):,}", flush=True)
    U, Pdg, Pdmax, ufr, ulast = U[ok], Pdg[ok], Pdmax[ok], ufr[ok], ulast[ok]
    NP2 = len(U); bad = np.zeros(NP2, bool); ok = ~bad
    lo_, hi_ = 1.0e2, 1.0e7
    for _ in range(60):
        mid = 0.5 * (lo_ + hi_)
        gg, _ = margins(U[ok], Pdg[ok], Pdmax[ok], ufr[ok], ulast[ok], mid)
        lo_, hi_ = (mid, hi_) if (gg <= 0).mean() < TARGET_PF else (lo_, mid)
    lam = 0.5 * (lo_ + hi_)
    g, mode = margins(U, Pdg, Pdmax, ufr, ulast, lam)
    S0 = float(g.std())
    np.save(f"resp_rbs_N{NP2}_clean.npy", dict(U=U, grid=GRID, Pd=Pdg, Pdmax=Pdmax,
            ufr=ufr, ulast=ulast, bad=bad), allow_pickle=True)
    fn = f"cache_rbs{TAG}_pf{TARGET_PF:.0e}_N{NP2}_seed0.npy"
    np.save(fn, dict(U=U, m=g, S0=S0, lam=lam, nbad=int(bad.sum()), mode=mode), allow_pickle=True)
    f = g <= 0
    print(f"\n{fn}  [{(time.time()-t0)/60:.1f} min]")
    print(f"  lambda = {lam/1e3:.2f} kN   N = {NP2:,}   NRV = {NRV}   theta_allow mean = {_THA_MED}")
    print(f"  Pf = {f.mean():.5f} ({int(f.sum())} failures)  counting CoV = "
          f"{1/np.sqrt(max(f.sum(),1)):.3f}")
    print(f"  modes: strength {int((mode[f]==1).sum())}, rupture {int((mode[f]==2).sum())}, "
          f"acceptance {int((mode[f]==3).sum())}")
    print(f"  u_fr finite (ruptured in window): {int(np.isfinite(ufr).sum()):,} "
          f"({np.isfinite(ufr).mean()*100:.1f} %);  U_FR_MIN guard fired on "
          f"{int((~np.isfinite(ufr) & (ulast < U_MAX - 2*DU)).sum())} samples")

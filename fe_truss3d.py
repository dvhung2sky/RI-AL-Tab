"""3D nonlinear 25-bar spatial transmission tower (OpenSeesPy corotTruss + Steel01).
Series-system limit state over members with yield + Euler-buckling capacity.
MULTI-RV redesign: RVs = 2 loads (vertical, lateral) + global E + global Fy + 8 canonical
area groups (the classic 25-bar member grouping, DISTINCT nominal sections) = 12 RVs.
E,Fy,A lognormal (CoV 0.05/0.08/0.05); loads normal. See fe_rv.py.

Phase 1 (OpenSees) evaluates g_true over a per-seed population in PARALLEL, caches to .npy;
Phase 2 (cheap) runs the AL arms on the cached g_true."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, time, multiprocessing as mp
import numpy as np
import fe_rv

# --- 25-bar spatial tower geometry (canonical) ---
NODES = np.array([
    [-37.5,   0.0, 200.0], [ 37.5,   0.0, 200.0],
    [-37.5,  37.5, 100.0], [ 37.5,  37.5, 100.0], [ 37.5, -37.5, 100.0], [-37.5, -37.5, 100.0],
    [-100.0, 100.0,  0.0], [ 100.0, 100.0,  0.0], [ 100.0,-100.0,  0.0], [-100.0,-100.0,  0.0],
], float)
MEMBERS = [
    (0, 1),
    (0, 3), (1, 2), (0, 4), (1, 5),
    (1, 3), (1, 4), (0, 2), (0, 5),
    (2, 5), (3, 4), (2, 3), (4, 5),
    (2, 9), (5, 6), (3, 8), (4, 7),
    (3, 6), (2, 7), (4, 9), (5, 8),
    (5, 9), (2, 6), (3, 7), (4, 8),
]
FIXED = [6, 7, 8, 9]; TOP_NODES = [0, 1]; NDOF = 3 * len(NODES)
_GEO = [(i, j, (NODES[j] - NODES[i]) / np.linalg.norm(NODES[j] - NODES[i]),
         float(np.linalg.norm(NODES[j] - NODES[i]))) for (i, j) in MEMBERS]
_Lm = np.array([L for (_, _, _, L) in _GEO])

# --- multi-RV model: material + section (8 canonical 25-bar area groups) ---
E0, FY0, HARD_B, KEFF, R_GYR = 1.0, 0.02, 0.05, 1.0, 7.5
MU, SIG_V, SIG_L = 1.0, 0.3, 0.5
N_LOAD = 2
GROUP = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7])
NOMINAL_A = np.array([1.0, 1.4, 1.2, 0.6, 0.8, 1.6, 1.3, 0.9])   # 8 DISTINCT nominal sections
A_NOM = NOMINAL_A[GROUP]                                          # (25,)
NRV = fe_rv.n_rv(N_LOAD, len(NOMINAL_A))                          # 12

# linear reference (nominal sections) for the calibration bracket + elastic sanity
_free = np.ones(NDOF, bool)
for n in FIXED: _free[3 * n:3 * n + 3] = False
_Kg = np.zeros((NDOF, NDOF))
for m, (i, j, c, L) in enumerate(_GEO):
    cc = np.outer(c, c) * (E0 * A_NOM[m] / L)
    dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
    _Kg[np.ix_(dofs, dofs)] += np.block([[cc, -cc], [-cc, cc]])
_Kff_inv = np.linalg.inv(_Kg[np.ix_(_free, _free)])
P_CR_NOM = (np.pi**2 * E0 * A_NOM * R_GYR**2) / (KEFF * _Lm)**2

def loads_from_U(U, lam):
    return lam * (MU + SIG_V * U[:, 0]), lam * (SIG_L * U[:, 1])    # P_vert, P_lat

def _member_cap(N, A_member, Fy, E):
    cap = A_member * Fy
    Pcr = (np.pi**2 * E * A_member * R_GYR**2) / (KEFF * _Lm)**2
    return np.where(N < 0, np.minimum(cap, Pcr), cap)

def linear_member_forces(P_vert, P_lat):
    f = np.zeros(NDOF)
    for n in TOP_NODES:
        f[3*n+2] += -P_vert; f[3*n+1] += P_lat
    u = np.zeros(NDOF); u[_free] = _Kff_inv @ f[_free]
    return np.array([(E0*A_NOM[m]/L) * (c @ (u[3*j:3*j+3] - u[3*i:3*i+3]))
                     for m, (i, j, c, L) in enumerate(_GEO)])

def opensees_member_forces(P_vert, P_lat, E, Fy, A_member, b=HARD_B, nstep=10):
    import openseespy.opensees as ops
    def _build():
        ops.wipe(); ops.model('basic', '-ndm', 3, '-ndf', 3)
        for n, (x, y, z) in enumerate(NODES): ops.node(n + 1, float(x), float(y), float(z))
        for n in FIXED: ops.fix(n + 1, 1, 1, 1)
        ops.uniaxialMaterial('Steel01', 1, float(Fy), float(E), float(b))
        for m, (i, j) in enumerate(MEMBERS, start=1):
            ops.element('corotTruss', m, i + 1, j + 1, float(A_member[m-1]), 1)
        ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1)
        for n in TOP_NODES: ops.load(n + 1, 0.0, float(P_lat), float(-P_vert))
        ops.system('BandGeneral'); ops.numberer('RCM'); ops.constraints('Plain')
    for nst, alg in [(nstep, 'Newton'), (4*nstep, 'LineSearch'), (12*nstep, 'Modified')]:
        _build(); ops.integrator('LoadControl', 1.0 / nst); ops.test('NormDispIncr', 1e-8, 100)
        if alg == 'LineSearch': ops.algorithm('NewtonLineSearch', 'tol', 0.8)
        elif alg == 'Modified': ops.algorithm('ModifiedNewton')
        else: ops.algorithm('Newton')
        ops.analysis('Static')
        if ops.analyze(nst) == 0:
            return np.array([ops.basicForce(m)[0] for m in range(1, len(MEMBERS) + 1)])
    return None

def margin_nl(P_vert, P_lat, E, Fy, A_member):
    N = opensees_member_forces(P_vert, P_lat, E, Fy, A_member)
    return None if N is None else float(np.min(_member_cap(N, A_member, Fy, E) - np.abs(N)))

def margin_lin_pop(U, lam):
    """Fast NOMINAL-section elastic margins (calibration bracket only)."""
    Pv, Pl = loads_from_U(U, lam); n = U.shape[0]; f = np.zeros((n, NDOF))
    for nd in TOP_NODES:
        f[:, 3*nd+2] += -Pv; f[:, 3*nd+1] += Pl
    u = np.zeros((n, NDOF)); u[:, _free] = f[:, _free] @ _Kff_inv.T
    N = np.empty((n, len(MEMBERS)))
    for m, (i, j, c, L) in enumerate(_GEO):
        N[:, m] = (E0*A_NOM[m]/L) * ((u[:, 3*j:3*j+3] - u[:, 3*i:3*i+3]) @ c)
    capY = A_NOM * FY0
    cap = np.where(N < 0, np.minimum(capY[None, :], P_CR_NOM[None, :]), capY[None, :])
    return np.min(cap - np.abs(N), axis=1)

def _margin_chunk(args):
    U_chunk, lam = args
    Pv, Pl = loads_from_U(U_chunk, lam)
    Ef, Fyf, A = fe_rv.material_section(U_chunk, N_LOAD, NOMINAL_A, GROUP)
    out = np.empty(len(U_chunk))
    for k in range(len(U_chunk)):
        g = margin_nl(float(Pv[k]), float(Pl[k]), E0*Ef[k], FY0*Fyf[k], A[k])
        out[k] = np.nan if g is None else g
    return out

_POOL = None; _NWORKERS = max(1, min(8, mp.cpu_count() - 2))
def _worker_init():
    dn = os.open(os.devnull, os.O_WRONLY); os.dup2(dn, 1); os.dup2(dn, 2)
def _get_pool():
    global _POOL
    if _POOL is None: _POOL = mp.Pool(_NWORKERS, initializer=_worker_init)
    return _POOL
def _close_pool():
    global _POOL
    if _POOL is not None: _POOL.close(); _POOL.join(); _POOL = None

def margins_pop(U, lam):
    chunks = np.array_split(U, _NWORKERS * 4)
    res = _get_pool().map(_margin_chunk, [(c, lam) for c in chunks])
    margins = np.concatenate(res); bad = ~np.isfinite(margins); nbad = int(bad.sum())
    if nbad:
        fin = margins[~bad]
        fill = min(float(fin.min()) if fin.size else -1.0, -1e-12)
        margins = np.where(bad, fill, margins)
    return margins, nbad

def calibrate_lambda(target_pf, n_calib, seed=12345, iters=8):
    rng = np.random.default_rng(seed); U = rng.standard_normal((n_calib, NRV))
    lo, hi = 1e-5, 1e4
    for _ in range(45):
        mid = np.sqrt(lo * hi)
        if float(np.mean(margin_lin_pop(U, mid) <= 0)) > target_pf: hi = mid
        else: lo = mid
    lam_lin = float(np.sqrt(lo * hi)); lo, hi = 0.3 * lam_lin, 2.0 * lam_lin
    for _ in range(iters):
        mid = np.sqrt(lo * hi); m, _ = margins_pop(U, mid)
        if float(np.mean(m <= 0)) > target_pf: hi = mid
        else: lo = mid
    lam = float(np.sqrt(lo * hi)); m, nbad = margins_pop(U, lam)
    return lam, float(np.mean(m <= 0)), nbad / n_calib, float(np.std(m))

TARGET_PF = 0.01; S0 = 1.0; CACHE = {}; N_POP = 40_000
def _cache_fn(seed): return f"cache_truss3d_mrv_pf{TARGET_PF:.0e}_seed{seed}.npy"
def precompute_seed(seed, lam, s0):
    fn = _cache_fn(seed)
    if os.path.exists(fn):
        return np.load(fn, allow_pickle=True).item()
    U = np.random.default_rng(seed).standard_normal((N_POP, NRV)); m, nb = margins_pop(U, lam)
    d = {"U": U, "m": m, "nbad": nb, "S0": s0, "lam": lam}
    np.save(fn, d, allow_pickle=True); return d

def make_world(seed, noise):
    d = CACHE[seed]; U, m = d["U"], d["m"]; g = m / d["S0"]
    if noise > 0:
        rng = np.random.default_rng(seed)
        z = rng.standard_t(4, size=len(U)) / np.sqrt(4/2)
        het = 1.0 + 0.5 * np.linalg.norm(U, axis=1) / np.sqrt(U.shape[1])
        y = g + noise * het * z
    else:
        y = g.copy()
    return U, y, g, float(np.mean(g <= 0))

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "validate":
    cond = np.linalg.cond(_Kg[np.ix_(_free, _free)])
    print(f"25-bar tower multi-RV | NRV={NRV} | {len(MEMBERS)} members, 8 groups A={NOMINAL_A}")
    print(f"  cond(Kff)={cond:.2e}" + ("  stable" if cond < 1e10 else "  MECHANISM!"))
    for Pv in (1e-4, 1e-5):
        Nl = linear_member_forces(Pv, 0.5*Pv)
        Nos = opensees_member_forces(Pv, 0.5*Pv, E0, 1e9, A_NOM, b=0.0, nstep=1)
        rel = np.abs(Nos - Nl).max() / (np.abs(Nl).max() + 1e-30)
        print(f"  elastic @P={Pv:.0e}: max|OS-lin|/max|lin| = {rel:.2e}")
    N1 = opensees_member_forces(0.02, 0.01, E0, FY0, A_NOM)
    N2 = opensees_member_forces(0.02, 0.01, E0, FY0, A_NOM)
    print(f"  determinism: {'OK' if (N1 is not None and np.array_equal(N1, N2)) else 'FAIL'}"
          f"  margin@(0.02,0.01)={margin_nl(0.02,0.01,E0,FY0,A_NOM):.4f}")
    _close_pool()

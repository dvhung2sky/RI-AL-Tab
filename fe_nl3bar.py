"""Nonlinear 3-bar truss (2D OpenSees: corotTruss geometric NL + Steel01 yielding).
Series-system limit state with yield + Euler-buckling capacity.
MULTI-RV redesign: random variables = 2 loads (Px,Py) + global E + global Fy + 2 area
groups (angled outer / vertical inner, DISTINCT nominal sections) = 6 RVs.
E,Fy,A lognormal (CoV 0.05/0.08/0.05); loads normal. See fe_rv.py."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time, multiprocessing as mp
import numpy as np
import fe_rv

NODES = np.array([[0., 0.], [-1., 1.], [0., 1.], [1., 1.]])   # 0=free, 1-3=supports
MEMBERS = [(0, 1), (0, 2), (0, 3)]
FIXED = [1, 2, 3]; LOAD_NODE = 0; NDOF = 2 * len(NODES)
_GEO = [(i, j, (NODES[j] - NODES[i]) / np.linalg.norm(NODES[j] - NODES[i]),
         float(np.linalg.norm(NODES[j] - NODES[i]))) for (i, j) in MEMBERS]

# ---- multi-RV model: material + section ----
E0, FY0, HARD_B, KEFF, R_GYR = 1.0, 0.02, 0.05, 1.0, 0.5
MU_Y, SIG_Y, SIG_X = 1.0, 0.3, 0.5
N_LOAD = 2
GROUP = np.array([0, 1, 0])                 # angled outer (0,1),(0,3) | vertical inner (0,2)
NOMINAL_A = np.array([1.2, 0.8])            # DISTINCT nominal areas per group
A_NOM = NOMINAL_A[GROUP]                     # (3,) nominal member areas
NRV = fe_rv.n_rv(N_LOAD, len(NOMINAL_A))     # 6
_Lm = np.array([L for (_, _, _, L) in _GEO])

# linear reference solver built with the NOMINAL sections (for calibration bracket)
_free = np.ones(NDOF, bool)
for n in FIXED: _free[2 * n:2 * n + 2] = False
_Kg = np.zeros((NDOF, NDOF))
for m, (i, j, c, L) in enumerate(_GEO):
    cc = np.outer(c, c) * (E0 * A_NOM[m] / L); d = [2*i, 2*i+1, 2*j, 2*j+1]
    _Kg[np.ix_(d, d)] += np.block([[cc, -cc], [-cc, cc]])
_Kff_inv = np.linalg.inv(_Kg[np.ix_(_free, _free)])
P_CR_NOM = (np.pi**2 * E0 * A_NOM * R_GYR**2) / (KEFF * _Lm)**2   # nominal buckling caps

def loads_from_U(U, lam):
    return lam * (SIG_X * U[:, 0]), lam * (MU_Y + SIG_Y * U[:, 1])   # Px, Py

def _member_cap(N, A_member, Fy, E):
    """Per-member capacity: yield A*Fy; compression min(yield, Euler pi^2 E A r^2/(KL)^2)."""
    cap = A_member * Fy
    Pcr = (np.pi**2 * E * A_member * R_GYR**2) / (KEFF * _Lm)**2
    return np.where(N < 0, np.minimum(cap, Pcr), cap)

def linear_forces(Px, Py):                   # nominal-section elastic forces (sanity/bracket)
    f = np.zeros(NDOF); f[2*LOAD_NODE] = Px; f[2*LOAD_NODE+1] = -Py
    u = np.zeros(NDOF); u[_free] = _Kff_inv @ f[_free]
    return np.array([(E0*A_NOM[m]/L)*(c @ (u[2*j:2*j+2]-u[2*i:2*i+2]))
                     for m, (i, j, c, L) in enumerate(_GEO)])

def opensees_forces(Px, Py, E, Fy, A_member, b=HARD_B, nstep=10):
    import openseespy.opensees as ops
    def _build():
        ops.wipe(); ops.model('basic', '-ndm', 2, '-ndf', 2)
        for n, (x, y) in enumerate(NODES): ops.node(n+1, float(x), float(y))
        for n in FIXED: ops.fix(n+1, 1, 1)
        ops.uniaxialMaterial('Steel01', 1, float(Fy), float(E), float(b))
        for m, (i, j) in enumerate(MEMBERS, 1):
            ops.element('corotTruss', m, i+1, j+1, float(A_member[m-1]), 1)
        ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1)
        ops.load(LOAD_NODE+1, float(Px), float(-Py))
        ops.system('BandGeneral'); ops.numberer('RCM'); ops.constraints('Plain')
    for nst, alg in [(nstep, 'Newton'), (4*nstep, 'LineSearch'), (12*nstep, 'Modified')]:
        _build(); ops.integrator('LoadControl', 1.0/nst); ops.test('NormDispIncr', 1e-8, 100)
        ops.algorithm('NewtonLineSearch', 'tol', 0.8) if alg == 'LineSearch' else \
            (ops.algorithm('ModifiedNewton') if alg == 'Modified' else ops.algorithm('Newton'))
        ops.analysis('Static')
        if ops.analyze(nst) == 0:
            return np.array([ops.basicForce(m)[0] for m in range(1, len(MEMBERS)+1)])
    return None

def margin_nl(Px, Py, E, Fy, A_member):
    N = opensees_forces(Px, Py, E, Fy, A_member)
    return None if N is None else float(np.min(_member_cap(N, A_member, Fy, E) - np.abs(N)))


# --------------- population / calibration / precompute --------------- #
def margin_lin_pop(U, lam):
    """Fast NOMINAL-section elastic margins over the population (calibration bracket only;
    loads vary, material/section at nominal). The nonlinear margins_pop carries the RVs."""
    Px, Py = loads_from_U(U, lam); n = len(U); f = np.zeros((n, NDOF))
    f[:, 2*LOAD_NODE] = Px; f[:, 2*LOAD_NODE+1] = -Py
    u = np.zeros((n, NDOF)); u[:, _free] = f[:, _free] @ _Kff_inv.T
    N = np.empty((n, len(MEMBERS)))
    for m, (i, j, c, L) in enumerate(_GEO):
        N[:, m] = (E0*A_NOM[m]/L) * ((u[:, 2*j:2*j+2] - u[:, 2*i:2*i+2]) @ c)
    capY = A_NOM * FY0
    cap = np.where(N < 0, np.minimum(capY[None, :], P_CR_NOM[None, :]), capY[None, :])
    return np.min(cap - np.abs(N), axis=1)

def _margin_chunk(args):
    U_chunk, lam = args
    Px, Py = loads_from_U(U_chunk, lam)
    Ef, Fyf, A = fe_rv.material_section(U_chunk, N_LOAD, NOMINAL_A, GROUP)
    out = np.empty(len(U_chunk))
    for k in range(len(U_chunk)):
        g = margin_nl(float(Px[k]), float(Py[k]), E0*Ef[k], FY0*Fyf[k], A[k])
        out[k] = np.nan if g is None else g
    return out

_POOL = None; _NWORKERS = max(1, min(8, mp.cpu_count() - 1))
def _wi():
    dn = os.open(os.devnull, os.O_WRONLY); os.dup2(dn, 1); os.dup2(dn, 2)
def margins_pop(U, lam):
    global _POOL
    if _POOL is None: _POOL = mp.Pool(_NWORKERS, initializer=_wi)
    res = _POOL.map(_margin_chunk, [(c, lam) for c in np.array_split(U, _NWORKERS*4)])
    m = np.concatenate(res); bad = ~np.isfinite(m); nb = int(bad.sum())
    if nb:
        fin = m[~bad]; fill = min(float(fin.min()) if fin.size else -1.0, -1e-12)
        m = np.where(bad, fill, m)
    return m, nb

def calibrate_lambda(target_pf, n_calib, seed=12345, iters=8):
    rng = np.random.default_rng(seed); U = rng.standard_normal((n_calib, NRV))
    lo, hi = 1e-5, 1e4
    for _ in range(45):
        mid = np.sqrt(lo*hi)
        if float(np.mean(margin_lin_pop(U, mid) <= 0)) > target_pf: hi = mid
        else: lo = mid
    ll = float(np.sqrt(lo*hi)); lo, hi = 0.3*ll, 2.0*ll
    for _ in range(iters):
        mid = np.sqrt(lo*hi); m, _ = margins_pop(U, mid)
        if float(np.mean(m <= 0)) > target_pf: hi = mid
        else: lo = mid
    lam = float(np.sqrt(lo*hi)); m, _ = margins_pop(U, lam)
    return lam, float(np.mean(m <= 0)), float(np.std(m))

TARGET_PF = 0.01; S0 = 1.0; CACHE = {}; N_POP = 40_000
def _cache_fn(seed): return f"cache_nl3bar_mrv_pf{TARGET_PF:.0e}_seed{seed}.npy"
def precompute_seed(seed, lam, s0):
    fn = _cache_fn(seed)
    if os.path.exists(fn):
        d = np.load(fn, allow_pickle=True).item(); return d
    U = np.random.default_rng(seed).standard_normal((N_POP, NRV)); m, nb = margins_pop(U, lam)
    d = {"U": U, "m": m, "nbad": nb, "S0": s0, "lam": lam}
    np.save(fn, d, allow_pickle=True); return d

def make_world(seed, noise):
    d = CACHE[seed]; U, m = d["U"], d["m"]; g = m / d["S0"]
    if noise > 0:
        rng = np.random.default_rng(seed)
        z = rng.standard_t(4, size=len(U)) / np.sqrt(4/2)
        het = 1.0 + 0.5 * np.linalg.norm(U, axis=1) / np.sqrt(U.shape[1])   # dim-normalized
        y = g + noise * het * z
    else:
        y = g.copy()
    return U, y, g, float(np.mean(g <= 0))

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "validate":
    cond = np.linalg.cond(_Kg[np.ix_(_free, _free)])
    print(f"3-bar multi-RV | NRV={NRV} | groups A={NOMINAL_A} | cond(Kff)={cond:.2e}"
          + ("  stable" if cond < 1e10 else "  MECHANISM!"))
    # elastic sanity: OpenSees at nominal props vs linear reference at tiny load
    for P in (1e-4, 1e-5):
        Nl = linear_forces(P, 0.5*P)
        Nos = opensees_forces(P, 0.5*P, E0, 1e9, A_NOM, b=0.0, nstep=1)
        rel = np.abs(Nos-Nl).max()/(np.abs(Nl).max()+1e-30)
        print(f"  elastic @P={P:.0e}: max|OS-lin|/max|lin| = {rel:.2e}")
    N1 = opensees_forces(0.05, 0.05, E0, FY0, A_NOM)
    N2 = opensees_forces(0.05, 0.05, E0, FY0, A_NOM)
    print(f"  determinism: {'OK' if np.array_equal(N1, N2) else 'FAIL'}")
    print(f"  member forces @(0.05,0.05):", np.round(N1, 4),
          "-> margin", round(margin_nl(0.05, 0.05, E0, FY0, A_NOM), 4))

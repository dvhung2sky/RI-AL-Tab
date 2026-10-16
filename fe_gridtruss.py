"""Fifth example: 3-D DOUBLE-LAYER GRID space truss (square-on-square offset), NX x NY.
Bottom node grid + offset raised top grid, joined by inverted-pyramid web diagonals.
Supports: 4 bottom corners pinned + 2 mid-span bottom-edge rollers. Elastic member forces,
capacity = YIELD & Euler BUCKLING with non-uniform slenderness (stocky chords / slender web).
MULTI-RV redesign: RVs = 2 load-direction inputs + global E + global Fy + 2 area groups
(chord / web, DISTINCT nominal sections) = 6 RVs. Forces at nominal sections (vectorized);
capacity carries random per-member A + global E,Fy (A CoV small -> force redistribution
negligible for this near-linear model). See fe_rv.py."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, time
import numpy as np
import fe_rv

NX = int(os.environ.get("GRID_NX", 8)); NY = int(os.environ.get("GRID_NY", 3))
DX = float(os.environ.get("GRID_DX", 2.0)); DY = float(os.environ.get("GRID_DY", 2.0))
H  = float(os.environ.get("GRID_H", 1.4))
RGYR_CHORD = float(os.environ.get("GRID_RGC", 0.30)); RGYR_WEB = float(os.environ.get("GRID_RGW", 0.09))
DTH = float(os.environ.get("GRID_DTH", 1.2)); TH0 = float(os.environ.get("GRID_TH0", 0.0))

NB = (NX + 1) * (NY + 1)
def b(i, j): return i * (NY + 1) + j
def t(i, j): return NB + i * NY + j
_nd = [(i * DX, j * DY, 0.0) for i in range(NX + 1) for j in range(NY + 1)]
_nd += [((i + 0.5) * DX, (j + 0.5) * DY, H) for i in range(NX) for j in range(NY)]
NODES = np.array(_nd, float); NN = len(NODES)

MEMBERS = []; MTYPE = []
def _add(a, c, ty): MEMBERS.append((a, c)); MTYPE.append(ty)
for i in range(NX):
    for j in range(NY + 1): _add(b(i, j), b(i + 1, j), 0)
for i in range(NX + 1):
    for j in range(NY): _add(b(i, j), b(i, j + 1), 0)
for i in range(NX - 1):
    for j in range(NY): _add(t(i, j), t(i + 1, j), 0)
for i in range(NX):
    for j in range(NY - 1): _add(t(i, j), t(i, j + 1), 0)
for i in range(NX):
    for j in range(NY):
        for (bi, bj) in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)):
            _add(t(i, j), b(bi, bj), 1)
MTYPE = np.array(MTYPE); NDOF = 3 * NN
_GEO = [(i, j, (NODES[j] - NODES[i]) / np.linalg.norm(NODES[j] - NODES[i]),
         float(np.linalg.norm(NODES[j] - NODES[i]))) for (i, j) in MEMBERS]

CORNERS = [b(0, 0), b(NX, 0), b(0, NY), b(NX, NY)]; MIDSPAN = [b(NX // 2, 0), b(NX // 2, NY)]
FIXED = {n: (1, 1, 1) for n in CORNERS}
for n in MIDSPAN: FIXED[n] = (0, 0, 1)
LOAD_NODES = [t(NX // 4, NY // 2), t(3 * NX // 4, NY // 2)]; LOAD_AXES = [0, 1]

# --- multi-RV model: material + section (2 groups: chord / web) ---
E0, FY0, KEFF, ULT = 1.0, 0.02, 1.0, 2.2
MU = float(os.environ.get("GRID_MU", 1.0))
N_LOAD = 2
GROUP = MTYPE                                        # chord=0, web=1
NOMINAL_A = np.array([1.5, 0.6])                     # DISTINCT nominal sections (chord | web)
A_NOM = NOMINAL_A[GROUP]
NRV = fe_rv.n_rv(N_LOAD, len(NOMINAL_A))             # 6
_Lm = np.array([L for (_, _, _, L) in _GEO])
_rg = np.where(MTYPE == 0, RGYR_CHORD, RGYR_WEB)
P_CR_NOM = (np.pi**2 * E0 * A_NOM * _rg**2) / (KEFF * _Lm)**2
_NLN = len(LOAD_NODES)

_fixed = np.zeros(NDOF, bool)
for n, fx in FIXED.items():
    for d in range(3):
        if fx[d]: _fixed[3 * n + d] = True
_free = ~_fixed
_Kg = np.zeros((NDOF, NDOF))
for m, (i, j, c, L) in enumerate(_GEO):
    cc = np.outer(c, c) * (E0 * A_NOM[m] / L)
    d = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
    _Kg[np.ix_(d, d)] += np.block([[cc, -cc], [-cc, cc]])
_Kff_inv = np.linalg.inv(_Kg[np.ix_(_free, _free)])

def loads_from_U(U, lam):
    """Two independent load-direction inputs: node A tilts in x-z by dir(U0), node B in y-z
    by dir(U1) -> member forces sinusoidal in the inputs (cusped, multimodal boundary)."""
    n = len(U); PX = np.zeros((n, _NLN)); PY = np.zeros((n, _NLN)); PZ = np.zeros((n, _NLN))
    M = lam * MU; th = TH0 + DTH * U[:, :_NLN]
    for k in range(_NLN):
        if LOAD_AXES[k] == 0: PX[:, k] = M * np.sin(th[:, k])
        else:                 PY[:, k] = M * np.sin(th[:, k])
        PZ[:, k] = -M * np.cos(th[:, k])
    return PX, PY, PZ

def _forces_nominal(U, lam):
    """Vectorized elastic member forces at NOMINAL sections, over the population."""
    PX, PY, PZ = loads_from_U(U, lam); n = len(U); f = np.zeros((n, NDOF))
    for k, nd in enumerate(LOAD_NODES):
        f[:, 3*nd] += PX[:, k]; f[:, 3*nd+1] += PY[:, k]; f[:, 3*nd+2] += PZ[:, k]
    u = np.zeros((n, NDOF)); u[:, _free] = f[:, _free] @ _Kff_inv.T
    N = np.empty((n, len(MEMBERS)))
    for m, (i, j, c, L) in enumerate(_GEO):
        N[:, m] = (E0*A_NOM[m]/L) * ((u[:, 3*j:3*j+3] - u[:, 3*i:3*i+3]) @ c)
    return N

def margins_pop(U, lam):
    """Elastic forces (nominal sections) + random capacity (per-member A, global E,Fy)."""
    N = _forces_nominal(U, lam)                                  # (n, M)
    Ef, Fyf, A = fe_rv.material_section(U, N_LOAD, NOMINAL_A, GROUP)
    E = (E0 * Ef)[:, None]; Fy = (FY0 * Fyf)[:, None]            # (n,1)
    capY = ULT * A * Fy                                          # (n, M)
    Pcr = (np.pi**2 * E * A * _rg[None, :]**2) / (KEFF * _Lm[None, :])**2
    cap = np.where(N < 0, np.minimum(capY, Pcr), capY)
    return np.min(cap - np.abs(N), axis=1), 0

def calibrate_lambda(target_pf, n_calib, seed=12345, iters=0):
    rng = np.random.default_rng(seed); U = rng.standard_normal((n_calib, NRV)); lo, hi = 1e-6, 1e5
    for _ in range(60):
        mid = np.sqrt(lo*hi); m, _ = margins_pop(U, mid)
        if float(np.mean(m <= 0)) > target_pf: hi = mid
        else: lo = mid
    lam = float(np.sqrt(lo*hi)); m, _ = margins_pop(U, lam)
    return lam, float(np.mean(m <= 0)), float(np.std(m))

TARGET_PF = float(os.environ.get("GRID_PF", 0.01)); S0 = 1.0; CACHE = {}
N_POP = int(os.environ.get("GRID_NPOP", 40_000))
def _cache_fn(seed):
    return (f"cache_gridtruss_mrv_{NX}x{NY}_rc{RGYR_CHORD:.2f}_rw{RGYR_WEB:.2f}"
            f"_dth{DTH:.1f}_pf{TARGET_PF:.0e}_seed{seed}.npy")
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
    nch = int((MTYPE == 0).sum()); nweb = int((MTYPE == 1).sum())
    print(f"3-D grid {NX}x{NY} multi-RV | NRV={NRV} | nodes={NN} members={len(MEMBERS)} "
          f"(chord {nch}, web {nweb})  groups A={NOMINAL_A}")
    print(f"  cond(Kff)={cond:.2e}" + ("  STABLE" if cond < 1e11 else "  MECHANISM!"))
    lam, pf, s0 = calibrate_lambda(0.01, 40000, iters=0)
    print(f"  quick calib @Pf=1e-2: lam={lam:.4g} pf={pf:.2e} s0={s0:.3g}")

"""A5-U (TabICL, conformal-UCB, BATCH=8, rep-3, median readout) NOISE-SCALE SWEEP.
Only sigma0 in make_world(seed, sigma0) changes: noise = sigma0*(1+0.5||x||)*t(4).
Reference Pf = mean(g_clean<=0) is noise-independent, so error is comparable across levels.
sigma0=0.15 should reproduce the 3-seed TabICL refs: fb 9.6 / 3-bar 22.5 / Warren 10.3 /
tower 31.9 / grid 8.1.  NOTE: torch 2.11 CPU-threading segfaults TabICL -> pinned to 1 thread
here (must be set before torch import). DEVICE env in {mps,cpu}; NOISES/SEEDS overridable."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time, warnings, numpy as np, torch
torch.set_num_threads(1)
warnings.filterwarnings("ignore")
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from tabicl import TabICLRegressor
import v1
from v1 import initial_doe, ALPHA
import fe_nl3bar as NB, fe_nlwarren as NW, fe_truss3d as TT, fe_gridtruss as GR

CK = "./tabicl-regressor-v2-20260212.ckpt"; CAND = int(os.environ.get("CAND", "6000"))
# BATCH = points acquired per surrogate refit. OOF_EVERY counts REFITS, so the conformal offset
# Qc is refreshed every OOF_EVERY*BATCH acquired points -- the two are coupled when BATCH is swept.
BATCH = int(os.environ.get("BATCH", "8")); OOF_EVERY = int(os.environ.get("OOF_EVERY", "5")); R = 3
DEVICE = os.environ.get("DEVICE", "mps")
NOISES = tuple(float(s) for s in os.environ.get("NOISES", "0.05,0.10,0.15,0.20,0.30").split(","))
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2").split(","))
REG = TabICLRegressor(n_estimators=1, device=DEVICE, model_path=CK, allow_auto_download=False, kv_cache=False)

def _oof(X, y, K=3):
    n = len(X); K = min(K, n); o = np.empty(n)
    for tr, va in KFold(K, shuffle=True, random_state=0).split(X):
        REG.fit(X[tr], y[tr]); o[va] = REG.predict(X[va])
    return np.abs(y - o)

def run_a5(pop, y, seed, rep):
    idx = initial_doe(seed); n = len(pop); target = v1.N0 + v1.BUDGET
    crng = np.random.default_rng(seed + 999 + 1000 * rep); q = None; refit = 0
    while len(idx) < target:
        X, yy = pop[idx], y[idx]
        if refit % OOF_EVERY == 0:
            r = _oof(X, yy); rho_tr = cKDTree(X).query(X, k=2)[0][:, 1] + 1e-9
            q = np.quantile(r / rho_tr, 1 - ALPHA)
        refit += 1; REG.fit(X, yy)
        cand = crng.choice(n, size=min(CAND, n), replace=False); Pc = pop[cand]; mu = REG.predict(Pc)
        rho_q = cKDTree(X).query(Pc, k=1)[0] + 1e-9; w = np.maximum(q * rho_q, 1e-12)
        amb = w * np.exp(-0.5 * (mu / w) ** 2); amb[np.isin(cand, idx)] = -1.0
        for _ in range(min(BATCH, target - len(idx))):
            j = int(np.argmax(amb))
            if amb[j] <= 0: break
            idx.append(int(cand[j]))
            d = np.linalg.norm(Pc - Pc[j], axis=1)
            amb *= 1.0 - np.exp(-0.5 * (d / rho_q[j]) ** 2); amb[j] = -1.0
    return idx

def med_pf(pop, idx, y):                      # A5-U median-quantile readout
    REG.fit(pop[idx], y[idx])
    o = REG.predict(pop, output_type=["mean", "quantiles"], alphas=[0.5])   # 2-elem -> dict
    med = o["quantiles"][:, 0]
    return float(np.mean(med <= 0))

def noop(s): pass
def nb_c(s):
    NB.TARGET_PF, NB.N_POP, NB.S0 = 0.01, 20_000, 0.003241
    d = np.load(f"cache_nl3bar_pf1e-02_seed{s}.npy", allow_pickle=True).item(); NB.CACHE[s] = (d["U"], d["m"], d["nbad"])
def nw_c(s):
    NW.TARGET_PF, NW.N_POP, NW.S0 = 0.01, 20_000, 0.004192
    d = np.load(f"cache_nlwarren_pf1e-02_seed{s}.npy", allow_pickle=True).item(); NW.CACHE[s] = (d["U"], d["m"], d["nbad"])
def tt_c(s):
    TT.TARGET_PF, TT.S0 = 0.01, 0.00358
    d = np.load(f"cache_truss3d_pf1e-02_seed{s}.npy", allow_pickle=True).item(); TT.CACHE[(0.01, s)] = (d["U"], d["m"], d["nbad"])
def gr_c(s):
    GR.S0 = 0.001872
    d = np.load(GR._cache_fn(s), allow_pickle=True).item(); GR.CACHE[s] = (d["U"], d["m"], d["nbad"])

EXAMPLES = [("four-branch", v1.make_world, noop, 20_000),
            ("3-bar",       NB.make_world, nb_c, 20_000),
            ("Warren",      NW.make_world, nw_c, 20_000),
            ("tower",       lambda s, n: TT.make_world(s, n), tt_c, 40_000),
            ("3D-grid",     GR.make_world, gr_c, 40_000)]

if __name__ == "__main__":
    print(f"A5-U (TabICL, {DEVICE}, 1-thread) NOISE SWEEP | sigma0={NOISES} | N=300 | seeds {SEEDS} "
          f"| rep{R}+median | BATCH=8\n", flush=True)
    for s0 in NOISES:
        print(f"--- sigma0 = {s0:.2f} " + "-" * 40, flush=True)
        for name, mw, load_cache, npop in EXAMPLES:
            v1.N_POP, v1.BUDGET = npop, 288
            errs = []; t0 = time.time()
            for seed in SEEDS:
                load_cache(seed)
                pop, y, g, rf = mw(seed, s0)
                pfs = [med_pf(pop, run_a5(pop, y, seed, r), y) for r in range(R)]
                errs.append(abs(float(np.mean(pfs)) - rf) / rf * 100)
            print(f"  s0={s0:.2f} {name:11s}: {np.mean(errs):5.1f} +/- {np.std(errs):4.1f}   "
                  f"per-seed={np.round(errs,1)}  [{time.time()-t0:.0f}s]", flush=True)
    print("DONE", flush=True)

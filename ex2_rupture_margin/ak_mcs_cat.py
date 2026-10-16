"""AK-MCS information ladder (plan section 9.1): canonical AK-MCS -- ak_mcs.py:run_akmcs, which stays
byte-identical -- with regime knowledge added as Kriging input columns. Saves the design (idx).

  RUNG  L0        plain AK-MCS; must reproduce the stored logs (c1-0.25 20.67, c2-0.5 16.45 mean D)
        L1        + regime flags: observed at evaluated points, a logistic gate's hard flags elsewhere
                  (refitted every iteration on the observed flags only)
        L2        + true regime flags for all N (the LEFM rule known)
        L2logr    + true flags and log(r / r*) for all N
        L3cap     Ex3: effective theta_u substituted into column 5, no flags (matches R4)
        L2strat   true regime flags for all N, one Kriging model per joint regime state (fitted on that
                  state's design points, on the plain inputs); a state with < NMIN_S = 10 design points
                  is predicted by the pooled L2 model instead (A8L2 paper, 2026-10-09, mock-review baseline)
        L3        + log(r / r*) ONLY, no flags -- the exact counterpart of ex3_cat.py's "R1 +log r"
                  readout, so the two surrogate families can be compared on the same 16 columns
                  (L2logr gives Kriging 17: flags AND capacity, which is a strictly stronger rung)

    cd conformal-reliability && CACHEF=ex3_variants/cache_rbs_<tag>_N49999_seed0.npy SEED=0 RUNG=L1 \
        python3 ex2_rupture_margin/ak_mcs_cat.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import ex3_cat as X                               # population, labels and gate; the TabICL model is unused
sys.path.insert(0, X.ROOT)
from ak_mcs import _kernel, N0

RUNG = os.environ.get("RUNG", "L0"); assert RUNG in ("L0", "L1", "L2", "L2logr", "L3", "L3cap", "L2strat")
SEED = X.SEED; BUD = X.NFE - N0
AKSUF = os.environ.get("AKSUF", "")      # "" keeps every existing filename; "_n100" separates budgets
RES = os.path.join(X.OUTD, f"ak_{X.TAG}{AKSUF}_{RUNG}_s{SEED}.json")


def feats(idx):
    if RUNG == "L0":
        return X.pop
    if RUNG == "L1":
        b = np.column_stack([X._gate_fit_proba(X.pop[idx], X.BTRUE[idx, j], X.pop, "lr") >= 0.5
                             for j in range(X.NCONN)]).astype(float)
        b[idx] = X.BTRUE[idx]                     # observed flags at evaluated points
        return np.c_[X.pop, b]
    if RUNG == "L3":
        return np.c_[X.pop, X.LOGR]               # capacity only: no flag, no gate
    if RUNG == "L3cap":                           # Ex3 only: the effective theta_u substituted in
        Xc = X.pop[:, :13].copy()                 # place, exactly as ex3_cat.py:339 does for R4
        Xc[:, 5] = np.log(X.c["theta_u"]); return Xc
    return np.c_[X.pop, X.BTRUE] if RUNG in ("L2", "L2strat") else np.c_[X.pop, X.BTRUE, X.LOGR]


def gp(Xf, idx):
    m = GaussianProcessRegressor(kernel=_kernel(Xf.shape[1]), alpha=1e-10, normalize_y=True,
                                 n_restarts_optimizer=0, random_state=0)
    return m.fit(Xf[idx], X.y[idx])


NMIN_S = 10
STATE = (X.BTRUE.astype(int) * (2 ** np.arange(X.BTRUE.shape[1]))).sum(1)   # joint regime state id


def predict(Xf, idx, return_std=True):
    """Pooled Kriging (every rung but L2strat), or one Kriging model per regime state (L2strat)."""
    out = gp(Xf, idx).predict(Xf, return_std=return_std)
    if RUNG != "L2strat":
        return out
    mu, sd = (out if return_std else (out, None))
    ia = np.asarray(idx)
    for k in np.unique(STATE):
        ik = ia[STATE[ia] == k]
        if len(ik) < NMIN_S: continue             # too few points in this state: keep the pooled L2 model
        m = GaussianProcessRegressor(kernel=_kernel(X.pop.shape[1]), alpha=1e-10, normalize_y=True,
                                     n_restarts_optimizer=0, random_state=0).fit(X.pop[ik], X.y[ik])
        sel = STATE == k
        if return_std: mu[sel], sd[sel] = m.predict(X.pop[sel], return_std=True)
        else: mu[sel] = m.predict(X.pop[sel])
    return (mu, sd) if return_std else mu


if __name__ == "__main__":
    if os.path.exists(RES):
        print(f"SKIP {os.path.basename(RES)}"); sys.exit(0)
    t0 = time.time()
    idx = list(np.random.default_rng(SEED).choice(X.N, size=N0, replace=False))   # ak_mcs.py:38-39
    for it in range(BUD):                          # ak_mcs.py:41-51, argmin U over the full population
        Xf = feats(idx); mu, sd = predict(Xf, idx)
        U = np.abs(mu) / np.maximum(sd, 1e-12); U[idx] = np.inf
        idx.append(int(np.argmin(U)))
    Xf = feats(idx); pred = predict(Xf, idx, return_std=False) <= 0
    idx = np.asarray(idx); known = np.zeros(X.N, bool); known[idx] = True
    out = dict(tag=X.TAG, rung=RUNG, seed=SEED, nfe=int(len(idx)), design_failures=int(X.truth[idx].sum()),
               AK=X.score(pred, known), t_total_s=round(time.time() - t0, 1))
    np.save(RES[:-5] + ".npy", idx); np.savez_compressed(RES[:-5] + "_pred.npz", pred=pred)
    open(RES, "w").write(json.dumps(out, indent=1) + "\n")
    print(f"RESULT ak {X.TAG} {RUNG} s{SEED}: D={out['AK']['D']:.2f} FP={out['AK']['FP']} FN={out['AK']['FN']} "
          f"sPf={out['AK']['sPf']:.2f} [{out['t_total_s']:.0f}s]", flush=True)

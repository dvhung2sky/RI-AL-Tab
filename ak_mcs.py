"""Canonical AK-MCS (Echard, Gayton & Lemaire 2011, Struct. Saf. 33:145-154).

Differs from this suite's A1 in three ways that matter in 13 dimensions:
  1. ANISOTROPIC (ARD) Kriging -- one length scale per variable. A1 uses a single isotropic RBF,
     which cannot down-weight the nine near-inactive dimensions.
  2. INTERPOLATING -- nugget ~ 0. A1 fits a WhiteKernel, so it smooths a limit state that is
     deterministic; AK-MCS as published interpolates the DoE exactly.
  3. STOPPING CRITERION min(U) >= 2 rather than a fixed budget. Reported both ways: at the
     criterion, and truncated to the common 300-FE budget so it is comparable with every other arm.

U(x) = |mu(x)| / sigma(x); the next point is argmin U over the Monte Carlo population.
Pf = fraction of the population with mu <= 0.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings, numpy as np; warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ex14pc")); sys.path.insert(0, os.path.join(HERE, "ex14fe"))
import v1

N0 = 12
BUD = int(os.environ.get("BUD", "288"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
KERN = os.environ.get("KERN", "matern")          # matern (AK-MCS default in most papers) | rbf


def _kernel(d):
    ls = np.ones(d)
    base = Matern(ls, [(1e-2, 1e3)] * d, nu=2.5) if KERN == "matern" else RBF(ls, [(1e-2, 1e3)] * d)
    return C(1.0, (1e-3, 1e3)) * base


def run_akmcs(pop, y, seed, budget=BUD, u_stop=2.0):
    """Returns (Pf at the stopping criterion, n_FE used, Pf at the full budget, min U history)."""
    d = pop.shape[1]
    rng = np.random.default_rng(seed)
    idx = list(rng.choice(len(pop), size=N0, replace=False))
    pf_stop, n_stop, pred_stop, hist = None, None, None, []
    for it in range(budget):
        gp = GaussianProcessRegressor(kernel=_kernel(d), alpha=1e-10, normalize_y=True,
                                      n_restarts_optimizer=0, random_state=0)
        gp.fit(pop[idx], y[idx])
        mu, sd = gp.predict(pop, return_std=True)
        U = np.abs(mu) / np.maximum(sd, 1e-12)
        U[idx] = np.inf
        hist.append(float(U.min()))
        if pf_stop is None and U.min() >= u_stop:
            pf_stop, n_stop, pred_stop = float((mu <= 0).mean()), len(idx), mu <= 0
        idx.append(int(np.argmin(U)))
    gp = GaussianProcessRegressor(kernel=_kernel(d), alpha=1e-10, normalize_y=True,
                                  n_restarts_optimizer=0, random_state=0)
    gp.fit(pop[idx], y[idx])
    mu_full = gp.predict(pop)
    pf_full = float((mu_full <= 0).mean())
    if pf_stop is None:
        pf_stop, n_stop, pred_stop = pf_full, len(idx), mu_full <= 0
    return pf_stop, n_stop, pf_full, hist, pred_stop, mu_full <= 0


if __name__ == "__main__":
    M = __import__(os.environ.get("MOD", "fe_yi2008"))
    dd = os.environ.get("DATADIR", "ex14fe")
    cache = os.path.join(HERE, dd, os.environ["CACHE"])
    M.CACHE[0] = np.load(cache, allow_pickle=True).item()
    for s in SEEDS:
        M.CACHE[s] = M.CACHE[0]
    pop, y, g, rf = M.make_world(SEEDS[0], 0.0)
    PFRB = float(os.environ.get("PFRB", "0.018326"))
    print(f"AK-MCS ({KERN}, ARD, interpolating) | N={len(pop):,} d={pop.shape[1]} | "
          f"Pf_RB={PFRB:.6f} | budget {N0+BUD} FE", flush=True)
    print(f"{'seed':>4} {'Pf@minU>=2':>11} {'n_FE':>5} {'err %':>7} | {'Pf@300FE':>9} {'err %':>7} {'s':>5}")
    A, B = [], []
    for s in SEEDS:
        t0 = time.time()
        ps, ns, pf, hist, qs, qf = run_akmcs(pop, y, s)
        A.append(ps); B.append(pf)
        print(f"{s:4d} {ps:11.6f} {ns:5d} {(ps-PFRB)/PFRB*100:+7.2f} | {pf:9.6f} "
              f"{(pf-PFRB)/PFRB*100:+7.2f} {time.time()-t0:5.0f}", flush=True)
        # set error against the EMPIRICAL failure set, comparable with A1-A11
        truth = g <= 0; nf = int(truth.sum())
        for nm, q, p in (("AK-MCS@stop", qs, ps), ("AK-MCS", qf, pf)):
            fp = int((q & ~truth).sum()); fn = int((~q & truth).sum())
            assert abs((fp-fn)/nf*100 - (p-rf)/rf*100) < 1e-6, "confusion identity"
            e = abs(p-rf)/rf*100; D = (fp+fn)/nf*100
            print(f"  {nm:20s}: Pf={p:.5f}  rel-err={e:7.2f}%  FP={fp:5d} FN={fn:5d} "
                  f"D={D:7.2f}% C={1-e/D if D>0 else 0:5.3f}   [{time.time()-t0:.0f}s]", flush=True)
    for nm, v in (("at min(U)>=2", A), ("at 300 FE", B)):
        e = (np.array(v) - PFRB) / PFRB * 100
        print(f"  AK-MCS {nm:14s}: mean |err| {np.abs(e).mean():6.2f} +/- {np.abs(e).std():5.2f}  "
              f"worst {np.abs(e).max():6.2f}  bias {e.mean():+6.2f}  n={len(v)}")

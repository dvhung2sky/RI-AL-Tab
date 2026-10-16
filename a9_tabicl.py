"""A9 = TabICL + CQR-conformal-UCB acquisition (Conformalized Quantile Regression).
Same A5 pipeline (TabICL surrogate, BATCH-8 + local penalisation, rep-3, MEDIAN readout),
but the acquisition width is the CQR band:  w = 0.5*(q95 - q05) + Q,
  Q = (1-ALPHA) quantile of the OOF conformity score E = max(q_lo - y, y - q_hi).
Unlike A5's near-uniform w = q*rho (one global quantile x distance), CQR keeps TabICL's OWN
input-dependent band SHAPE (wide where the model is unsure, narrow where confident) and only
ADDS a constant conformal offset Q for coverage. Tests whether an adaptive, coverage-corrected
band beats both the uniform-conformal (A5) and the raw-native (A7) widths."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings, numpy as np, torch
torch.set_num_threads(1)
warnings.filterwarnings("ignore")
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
import v1
from v1 import initial_doe, ALPHA
from a5u_noise import REG, med_pf, R, CAND, BATCH, OOF_EVERY

A_LO, A_HI = ALPHA / 2, 1 - ALPHA / 2

KOOF = int(os.environ.get("KOOF", "3"))     # OOF folds behind the conformal offset Qc
# Local-penalisation length-scale multiplier. The batch repulsion is 1 - exp(-0.5*(d/(LP*rho))^2)
# with rho the candidate's 1-NN distance to the design. LP=0 disables repulsion entirely, so the
# batch becomes the plain top-BATCH of the ambiguity score.
LP = float(os.environ.get("LP", "1.0"))
# QCOFF=1 zeroes the conformal offset, leaving the RAW native CQR band 0.5*(q95-q05) as the
# acquisition width -- the ablation that removes conformal calibration but keeps the band shape.
QCOFF = bool(int(os.environ.get("QCOFF", "0")))

def _oof_cqr_Q(X, y, K=KOOF):
    """OOF CQR conformity offset Q = quantile_{1-ALPHA} of E = max(q_lo - y, y - q_hi)."""
    n = len(X); K = min(K, n); lo = np.empty(n); hi = np.empty(n)
    for tr, va in KFold(K, shuffle=True, random_state=0).split(X):
        REG.fit(X[tr], y[tr])
        o = REG.predict(X[va], output_type=["mean", "quantiles"], alphas=[A_LO, A_HI])
        Q = o["quantiles"]; lo[va] = Q[:, 0]; hi[va] = Q[:, 1]
    E = np.maximum(lo - y, y - hi)
    return float(np.quantile(E, 1 - ALPHA))

def run_a9(pop, y, seed, rep):
    idx = initial_doe(seed); n = len(pop); target = v1.N0 + v1.BUDGET
    crng = np.random.default_rng(seed + 999 + 1000 * rep); Qc = 0.0; refit = 0
    while len(idx) < target:
        X, yy = pop[idx], y[idx]
        if refit % OOF_EVERY == 0:
            Qc = 0.0 if QCOFF else _oof_cqr_Q(X, yy)                # conformal offset (recomputed)
        refit += 1; REG.fit(X, yy)
        cand = crng.choice(n, size=min(CAND, n), replace=False); Pc = pop[cand]
        o = REG.predict(Pc, output_type=["mean", "quantiles"], alphas=[A_LO, A_HI])
        mu = o["mean"]; ql, qh = o["quantiles"][:, 0], o["quantiles"][:, 1]
        w = np.maximum(0.5 * (qh - ql) + Qc, 1e-12)                # CQR band: adaptive shape + offset
        rho_q = cKDTree(X).query(Pc, k=1)[0] + 1e-9                # length scale for diversity only
        amb = w * np.exp(-0.5 * (mu / w) ** 2); amb[np.isin(cand, idx)] = -1.0
        for _ in range(min(BATCH, target - len(idx))):
            j = int(np.argmax(amb))
            if amb[j] <= 0: break
            idx.append(int(cand[j]))
            if LP > 0:
                d = np.linalg.norm(Pc - Pc[j], axis=1)
                amb *= 1.0 - np.exp(-0.5 * (d / (LP * rho_q[j])) ** 2)
            amb[j] = -1.0
    return idx

def a9_pf(pop, y, seed):
    return float(np.mean([med_pf(pop, run_a9(pop, y, seed, r), y) for r in range(R)]))

if __name__ == "__main__":
    v1.N_POP, v1.BUDGET = 20_000, 288
    pop, yv, g, rf = v1.make_world(0, 0.15)
    p = a9_pf(pop, yv, 0)
    print(f"A9 four-branch noisy seed0: Pf={p:.4f} ref={rf:.4f} err={abs(p-rf)/rf*100:.1f}%")

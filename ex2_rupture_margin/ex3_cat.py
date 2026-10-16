"""Ex3 regime-aware acquisition -- a trimmed copy of ex3_phys.py, which stays byte-identical.

Plan: A11/Submission/intent/mode-aware-acquisition-plan.md, sections 3 and 5.

  ACQ        a8 | cat         a8 = A8 exactly (TabICL + U-function, isotropic repulsion)
  GATE       const | oracle | lr | clf | base
             const = one regime, scored through the tau-link (checks that the link nests A8)
  LABELS     true | placebo | censored     placebo = a learnable regime with no physics (plan section 6, arm 6);
             censored = only the governing connection's flag is observed (a model that stops at the first
             fracture, plan 9.4); each gate trains on its observed points and imputes the rest
  QC         none | global | regime     conformal offset added to the half-band (Mondrian per state)
  BATCHRULE  rep | regrep | deficit     A8 repulsion | within-regime repulsion | + cumulative deficit
  BATCHN     batch size, default 2 (the operating point); NFE overrides 300 for smoke tests only
  DESIGN     <npy>: readout-only on a stored design; NPREF truncates it, LADDER=1 adds the 4.3 rungs
  CHECK      <npy>: assert idx == stored[:len(idx)] after every refit (acceptance test 2)
  RO_GATE    gate of the headline cat+mem readout (lr | clf), fixed by plan section 4.5
  POPF       <npz>: a generic population instead of an Ex3 cache (plan 9.2): U, g, B (regime flags,
             N x NCONN), LOGR (signed switch distance); e.g. ex2_rupture_margin/snap/pop_snap_w0.4_N50000.npz
  FEAT       none | logr | thueff   knowledge given to the surrogate as inputs, for every fit (RQ1 rungs):
             logr appends log(r / r*) per connection (the fracture criterion's form); thueff puts the
             effective theta_u in column 5 and drops the appended columns (L3, phys2's premise)

Regime flags b (per-connection LEFM state) are read only at evaluated points, except GATE=oracle.
Readouts always use the TRUE flags at evaluated points, so every arm is scored the same way.

    cd conformal-reliability && CACHEF=ex3_variants/cache_rbs_c1_fad0.3-0.25_t0.127_pf3e-03_N49999_seed0.npy \\
        SEED=0 ACQ=cat GATE=lr ARM=M2 python3 ex2_rupture_margin/ex3_cat.py
"""
import itertools, json, os, re, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("DEVICE", "cpu")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr, ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "ex14fe")):
    sys.path.insert(0, p)
E = os.environ.get
ACQ, GATE, QC, BR = E("ACQ", "cat"), E("GATE", "lr"), E("QC", "none"), E("BATCHRULE", "regrep")
LABELS = E("LABELS", "true")
assert ACQ in ("a8", "cat") and GATE in ("const", "oracle", "lr", "clf", "base") and LABELS in ("true", "placebo", "censored")
assert QC in ("none", "global", "regime") and BR in ("rep", "regrep", "deficit")
SEED = int(E("SEED", "0")); NFE = int(E("NFE", "300")); NMIN = int(E("NMIN", "10"))   # NMIN, NFE: smoke tests only
CHUNK = 50_000                                   # TabICL changes predictions at >= 65,536 query rows
Z = 1.6448536269514722                           # Phi^-1(0.95): 0.5*(q95 - q05) is Z standard deviations
POPF, CACHEF = E("POPF"), E("CACHEF")
if POPF:                                         # generic population: regimes come from the file
    Pz = np.load(POPF if os.path.isabs(POPF) else os.path.join(ROOT, POPF))
    TAG = os.path.basename(POPF)[:-4]; NCONN = Pz["B"].shape[1]; P_BR, KAP = 1.0, None
else:
    TAG = re.search(r"cache_rbs_(.+)_N\d+_seed0", CACHEF).group(1)
    NCONN = 2 if TAG.startswith("c2") else 1
    _f = re.search(r"fad([\d.]+)-([\d.]+)", TAG)
    P_BR, KAP = map(float, _f.groups()) if _f else (0.0, 1.0)
ARM = E("ARM", f"{ACQ}-{GATE}-{QC}-{BR}"); DESIGN, CHECK = E("DESIGN"), E("CHECK")
LADDER = E("LADDER") == "1"; RO_GATE = E("RO_GATE", "lr"); assert RO_GATE in ("lr", "clf")
OUTD = os.path.join(HERE, "ex3cat_runs"); os.makedirs(OUTD, exist_ok=True)

import v1
M = __import__("fe_rbs")
if POPF:
    c = {}; pop, g = Pz["U"], Pz["g"]; y = g.copy()
else:
    c = np.load(os.path.join(ROOT, "ex14fe", CACHEF), allow_pickle=True).item()
    M.CACHE[SEED] = c
    pop, y, g, rf = M.make_world(SEED, 0.0)
N = len(pop); truth = g <= 0; nf = int(truth.sum())
v1.N_POP, v1.BUDGET = N, NFE - v1.N0
from a5u_noise import REG, CAND, OOF_EVERY
from a9_tabicl import _oof_cqr_Q, LP
from v1 import initial_doe, ALPHA
A_LO, A_HI = ALPHA / 2, 1 - ALPHA / 2
T = lambda t: t / (1.0 + np.abs(t))
BATCH = int(E("BATCHN", "2"))
assert (LP, CAND, OOF_EVERY) == (1.0, 6000, 5)


def labels():
    """Per-connection LEFM flags (make_rbs_variant.py:44-56), log(r / r*) and the governing connection.
    The stored effective theta_u must be reproduced exactly before anything uses the flags."""
    if POPF:
        return Pz["B"].astype(bool), Pz["LOGR"].astype(float), np.zeros(N, int)
    U = c["U"]; fyr = M._ln(U[:, 0], M._COV["fy"]); th = M.A.RBS["theta_u"]
    tus = [th * M._ln(U[:, 5], M._COV["thu"])] + ([th * M._ln(U[:, 13], M._COV["thu"])] if NCONN == 2 else [])
    B, LR = np.zeros((N, NCONN), bool), np.zeros((N, NCONN))
    for j in range(NCONN if P_BR > 0 else 0):
        c0 = (14 if NCONN == 2 else 13) + 2 * j
        r = fyr * np.sqrt(M._ln(U[:, c0], 0.5)) / M._ln(U[:, c0 + 1], 0.3)
        rs = np.quantile(r, 1 - P_BR); B[:, j] = r > rs; LR[:, j] = np.log(r / rs)
        tus[j] = np.where(B[:, j], KAP * tus[j], tus[j])
    tus = np.vstack(tus)
    if "theta_u" in c:
        assert np.abs(tus.min(0) - c["theta_u"]).max() == 0.0, "regime flags do not reproduce theta_u"
    return B, LR, tus.argmin(0)


def placebo_flags():
    """A learnable regime with no physics: the top 30 % along a fixed random direction in the two
    measured-inert inputs E_s (x4) and t_doubler (x7), fe_rbs.py NOTE 2. One flag per connection."""
    s = pop[:, [4, 7]] @ np.random.default_rng(2026).standard_normal((2, NCONN))
    return s > np.quantile(s, 0.7, axis=0)


BTRUE, LOGR, GOVJ = labels()
GOV = BTRUE[np.arange(N), GOVJ]                  # governing connection brittle: reporting only
BREG = placebo_flags() if LABELS == "placebo" else BTRUE   # the regime the acquisition is told about
FEAT = E("FEAT", "none"); assert FEAT in ("none", "logr", "thueff")
if FEAT == "logr":
    pop = np.c_[pop, LOGR]
elif FEAT == "thueff":
    pop = pop[:, :13].copy(); pop[:, 5] = np.log(c["theta_u"])
STATES = np.array(list(itertools.product((0.0, 1.0), repeat=NCONN))); K = len(STATES)
NEAR = (np.abs(LOGR) < 0.1).any(1)               # within 0.1 in log r of a switch (logging only)
sidx = lambda b: (np.asarray(b, int) * 2 ** np.arange(NCONN)[::-1]).sum(-1)   # state index of flags

_CLF = []


def _gate_fit_proba(Xtr, btr, Xte, kind):
    """P(flag = 1) for one connection. A single-class training set returns its observed share."""
    if btr.min() == btr.max():
        return np.full(len(Xte), float(btr.mean()))
    if kind == "clf":
        if not _CLF:
            from tabicl import TabICLClassifier
            _CLF.append(TabICLClassifier(n_estimators=1, device="cpu", kv_cache=False, allow_auto_download=False,
                                         model_path=os.path.join(ROOT, "tabicl-classifier-v2-20260212.ckpt")))
        m = _CLF[0]; m.fit(Xtr, btr.astype(int))
    else:
        m = LogisticRegression(max_iter=5000).fit(Xtr, btr.astype(int))
    return m.predict_proba(Xte)[:, list(m.classes_).index(1)]


def gate_eta(Xtr, btr, kind):
    """Laplace-smoothed 3-fold out-of-fold error of the gate: the floor on membership probabilities."""
    err = 0
    for tr, va in KFold(3, shuffle=True, random_state=0).split(Xtr):
        err += int(((_gate_fit_proba(Xtr[tr], btr[tr], Xtr[va], kind) >= 0.5) != btr[va]).sum())
    return (err + 1) / (len(btr) + 2)


def observed(idx):
    """(len(idx), NCONN) mask of the flags observed at evaluated points: all of them, or under
    LABELS=censored only the governing connection's."""
    idx = np.asarray(idx)
    return (GOVJ[idx][:, None] == np.arange(NCONN)) if LABELS == "censored" else np.ones((len(idx), NCONN), bool)


def flags_at(idx, Bsrc, kind):
    """Flags at evaluated points: observed ones from Bsrc, the rest imputed by that connection's gate,
    trained on its observed points only (a connection never observed is imputed ductile)."""
    idx = np.asarray(idx); b = Bsrc[idx].copy(); ob = observed(idx)
    for j in range(NCONN):
        if (~ob[:, j]).any():
            tr = idx[ob[:, j]]
            b[~ob[:, j], j] = (_gate_fit_proba(pop[tr], Bsrc[tr, j], pop[idx[~ob[:, j]]], kind) >= 0.5) if len(tr) else False
    return b


def pconn(idx, rows, kind, Bsrc, eta=None):
    """(len(rows), NCONN) P(flag = 1) per connection from the flags Bsrc observed at idx only.
    oracle reads Bsrc everywhere (white-box bound); base is the observed share; lr/clf are floored."""
    idx = np.asarray(idx); ob = observed(idx)
    out = np.empty((len(rows), NCONN))
    for j in range(NCONN):
        tr = idx[ob[:, j]]; b = Bsrc[tr, j]
        if kind == "oracle":
            out[:, j] = Bsrc[rows, j]
        elif kind == "base":
            out[:, j] = b.mean()
        else:
            gk = "clf" if kind == "clf" else "lr"
            e = gate_eta(pop[tr], b, gk) if eta is None else eta[j]
            out[:, j] = np.clip(_gate_fit_proba(pop[tr], b, pop[rows], gk), e, 1 - e)
    return out


def pstate(pc):
    """(n, K) probability of each joint regime state, per-connection gates independent."""
    return np.prod(np.where(STATES[None] == 1, pc[:, None, :], 1 - pc[:, None, :]), axis=2)


def link(mu, sd, P):
    """Regime-mixture misclassification probability tau = min(pi, 1 - pi) through A8's link
    a = exp(-beta^2 / (2 Z^2)), beta = -Phi^-1(tau). Both tails are summed directly, so neither
    saturates; at K = 1 with sd = sig / Z this is exp(-0.5 (mu / sig)^2), A8's amb. Also returns the
    law-of-total-variance split pi (1 - pi) = W + G and the per-state tau_k of the deficit rule."""
    pk, qk = ndtr(-mu / sd), ndtr(mu / sd)
    pi, qi = (P * pk).sum(1), (P * qk).sum(1)
    beta = -ndtri(np.maximum(np.minimum(pi, qi), 1e-300))
    return (np.exp(-beta ** 2 / (2 * Z * Z)), (P * pk * qk).sum(1), (P * (pk - pi[:, None]) ** 2).sum(1),
            np.minimum(pk, qk))


def qc_regime(Xb, yy, key):
    """Per-state CQR offset (Mondrian): 3-fold out-of-fold conformity scores of the regime-column
    model, split by each evaluated point's own state. A state with < NMIN points takes the pooled one."""
    lo, hi = np.empty(len(yy)), np.empty(len(yy))
    for tr, va in KFold(3, shuffle=True, random_state=0).split(Xb):
        REG.fit(Xb[tr], yy[tr])
        q = REG.predict(Xb[va], output_type=["mean", "quantiles"], alphas=[A_LO, A_HI])["quantiles"]
        lo[va], hi[va] = q[:, 0], q[:, 1]
    Es = np.maximum(lo - yy, yy - hi); pooled = np.quantile(Es, 1 - ALPHA)
    return np.array([np.quantile(Es[key == k], 1 - ALPHA) if (key == k).sum() >= NMIN else pooled
                     for k in range(K)])


def acquire(check=None):
    idx = initial_doe(SEED); crng = np.random.default_rng(SEED + 999); refit = 0
    Qg, Qk, eta, log, t_on = 0.0, np.zeros(K), None, [], None
    while len(idx) < NFE:
        X, yy = pop[idx], y[idx]
        b = flags_at(idx, BREG, "clf" if GATE == "clf" else "lr") if LABELS == "censored" and GATE != "oracle" else BREG[idx]
        # guard: the mixture runs only once every regime column has >= NMIN evaluated points in each
        # value; before that the refit is the K = 1 path, which is A8 exactly (plan 3.4)
        mix = ACQ == "cat" and GATE != "const" and all(
            min(b[:, j].sum(), (~b[:, j]).sum()) >= NMIN for j in range(NCONN))
        if mix and t_on is None:
            t_on = len(idx)
        Xb = np.c_[X, b] if mix else X
        if refit % OOF_EVERY == 0 or (mix and eta is None):
            if QC == "global":
                Qg = _oof_cqr_Q(Xb, yy)
            if mix and QC == "regime":
                Qk = qc_regime(Xb, yy, sidx(b))
            if mix:
                ob = observed(idx)
                eta = ([gate_eta(X[ob[:, j]], BREG[idx][ob[:, j], j], "clf" if GATE == "clf" else "lr") for j in range(NCONN)]
                       if GATE in ("lr", "clf") else [0.0] * NCONN)
        refit += 1
        cand = crng.choice(N, size=CAND, replace=False); Pc = pop[cand]
        rho = cKDTree(X).query(Pc, k=1)[0] + 1e-9
        REG.fit(Xb, yy)
        if mix:
            P = pstate(pconn(idx, cand, GATE, BREG, eta))
            mu, sd = np.empty((CAND, K)), np.empty((CAND, K))
            for k, s in enumerate(STATES):           # one predict call per state: position-deterministic
                o = REG.predict(np.c_[Pc, np.tile(s, (CAND, 1))], output_type=["mean", "quantiles"],
                                alphas=[A_LO, A_HI])
                mu[:, k] = o["mean"]
                sd[:, k] = np.maximum(0.5 * (o["quantiles"][:, 1] - o["quantiles"][:, 0]) + Qg + Qk[k], 1e-12) / Z
            amb, W, G, tk = link(mu, sd, P)
        else:
            o = REG.predict(Pc, output_type=["mean", "quantiles"], alphas=[A_LO, A_HI])
            sig = np.maximum(0.5 * (o["quantiles"][:, 1] - o["quantiles"][:, 0]) + Qg, 1e-12)
            P = np.ones((CAND, 1))
            if ACQ == "cat" and GATE == "const":     # K = 1 through the tau-link
                amb = link(o["mean"][:, None], sig[:, None] / Z, P)[0]
            else:                                    # A8 exactly (ex3_phys.py:96-97)
                r2 = (o["mean"] / sig) ** 2; amb = np.exp(-0.5 * r2); new = ~np.isin(cand, idx)
                if not (amb[new] > 0).any():         # every new candidate underflows: ex3_phys.py admits no pick and
                    amb = np.exp(-0.5 * (r2 - r2[new].min()))   # loops forever (snap s0, n = 242). Same ranking,
        amb[np.isin(cand, idx)] = -1.0               # rescaled; never fires where ex3_phys.py terminates (all of Ex3)
        hard = P.argmax(1)
        if mix:
            nk = np.bincount(sidx(b), minlength=K).astype(float)
            q = (P * tk).sum(0); q = q / q.sum()     # each state's share of expected misclassification
        picks = []
        for _ in range(min(BATCH, NFE - len(idx))):
            if mix and BR == "deficit":              # the state most owed picks goes first
                el = np.where(hard == int(np.argmax(q * len(idx) - nk)), amb, -np.inf)
                j = int(np.argmax(el)) if el.max() > 0 else int(np.argmax(amb))
            else:
                j = int(np.argmax(amb))
            if amb[j] <= 0: break
            pick = int(cand[j]); pt = Pc[j]; rp = rho[j]; amb[j] = -1.0
            idx.append(pick); picks.append(j)
            if mix:
                nk[hard[j]] += 1
            # within-regime repulsion: candidates repel only as far as they share the pick's regime;
            # s = 1 gives A8's own repulsion (ex3_phys.py:122), applied to every entry as there
            s = P @ P[j] if (mix and BR != "rep") else 1.0
            amb = amb * (1.0 - s * np.exp(-0.5 * (np.linalg.norm(Pc - pt, axis=1) / rp) ** 2))
        rec = dict(n=len(idx) - len(picks), mix=bool(mix), pick_state=[int(sidx(BREG[cand[j]])) for j in picks],
                   pick_brittle=[bool(GOV[cand[j]]) for j in picks])
        if mix:
            ok = hard == sidx(BREG[cand]); nr = NEAR[cand]
            rec.update(acc=round(float(ok.mean()), 4), acc_near=round(float(ok[nr].mean()), 4) if nr.any() else None,
                       SW=round(float(W.sum()), 3), SG=round(float(G.sum()), 3),
                       pick_GgtW=[bool(G[j] > W[j]) for j in picks], eta=[round(float(e), 4) for e in eta],
                       disagree=int(((mu <= 0).any(1) & (mu > 0).any(1)).sum()))
        log.append(rec)
        if check is not None:
            assert [int(v) for v in idx] == [int(v) for v in check[:len(idx)]], f"design diverged at n={len(idx)}"
    return np.asarray(idx), log, t_on


def pred_q(Xtr, ytr, Xtes, alphas=(0.5,)):
    """Fit once, then population quantiles for each matrix in chunks (ex3_phys.py:66-68)."""
    REG.fit(Xtr, ytr)
    return [np.concatenate([REG.predict(Xe[i:i + CHUNK], output_type=["mean", "quantiles"], alphas=list(alphas))
                            ["quantiles"] for i in range(0, len(Xe), CHUNK)]) for Xe in Xtes]


def score(pred, known):
    fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
    fno = int((~pred & truth & ~known).sum())
    return dict(D=round((fp + fn) / nf * 100, 4), FP=fp, FN=fn, sPf=round((fp - fn) / nf * 100, 4),
                FN_brittle=int((~pred & truth & GOV).sum()), FN_ductile=int((~pred & truth & ~GOV).sum()),
                FP_brittle=int((pred & ~truth & GOV).sum()), FP_ductile=int((pred & ~truth & ~GOV).sum()),
                recall_off=round(1 - fno / max(int((truth & ~known).sum()), 1), 4))


def readouts(idx, known):
    """Predicted failure labels (N bools) per readout. Tg+mem is label-blind, the published comparator;
    on FAD variants cat+mem is the regime-mixture plug-in classifier, centred on the predictive median
    so that one regime reproduces Tg+mem exactly. RO=blind stops after Tg+mem (prefix curves);
    LADDER=1 adds the plan's section 4.3 rungs."""
    r, yT = {}, T(y[idx])
    mg = pred_q(pop[idx], yT, [pop])[0][:, 0]
    r["Tg"] = mg <= 0; r["Tg+mem"] = np.where(known, truth, mg <= 0)
    if E("RO") == "blind":
        return r
    if E("RO") != "min":                             # RO=min: Tg+mem and cat+mem only (Gate A)
        r["raw g"] = pred_q(pop[idx], y[idx], [pop])[0][:, 0] <= 0
    if P_BR == 0:
        return r
    bq = flags_at(idx, BTRUE, RO_GATE) if LABELS == "censored" else BTRUE[idx]   # censored: imputed columns
    Q = pred_q(np.c_[pop[idx], bq], yT, [np.c_[pop, np.tile(s, (N, 1))] for s in STATES], (A_LO, 0.5, A_HI))
    med = np.column_stack([q[:, 1] for q in Q])
    sd = np.column_stack([np.maximum(0.5 * (q[:, 2] - q[:, 0]), 1e-12) / Z for q in Q])
    pk, qk = ndtr(-med / sd), ndtr(med / sd)
    for gk in (() if E("RO") == "min" else ("oracle",)) + (RO_GATE,) + (("clf",) if LADDER and RO_GATE != "clf" else ()):
        P = pstate(pconn(idx, np.arange(N), gk, BTRUE))
        name = "cat+mem" if gk == RO_GATE else f"cat+mem {gk}"
        r[name] = np.where(known, truth, (P * pk).sum(1) >= (P * qk).sum(1))
        if LADDER:                                   # hard flags as a feature: the critic's R3 rung
            r[f"hard+mem {gk}"] = np.where(known, truth, med[np.arange(N), P.argmax(1)] <= 0)
    if LADDER:
        f = lambda X: np.where(known, truth, pred_q(X[idx], yT, [X])[0][:, 0] <= 0)
        r["R1 +log r"] = f(np.c_[pop, LOGR])
        if "theta_u" in c:
            X4 = pop[:, :13].copy(); X4[:, 5] = np.log(c["theta_u"]); r["R4 theta_u_eff"] = f(X4)
        if NCONN == 2:
            r["gov flag"] = f(np.c_[pop, GOV])
    return r


if __name__ == "__main__":
    t0 = time.time()
    if DESIGN:
        idx = np.load(DESIGN if os.path.isabs(DESIGN) else os.path.join(ROOT, DESIGN))[:int(E("NPREF", NFE))]
        name = ("ro_" + os.path.basename(DESIGN)[:-4] + (f"_n{len(idx)}" if len(idx) < NFE else "")
                + ("_ladder" if LADDER else "") + ("_blind" if E("RO") == "blind" else "")
                + ("" if RO_GATE == "lr" else f"_{RO_GATE}"))
        RES = os.path.join(OUTD, name + ".json"); log, t_on = [], None
    else:
        RES = os.path.join(OUTD, f"v_{TAG}_{ARM}_s{SEED}.json")
    if os.path.exists(RES):
        print(f"SKIP {os.path.basename(RES)}"); sys.exit(0)
    if not DESIGN:
        idx, log, t_on = acquire(np.load(CHECK) if CHECK else None)
        assert len(idx) == NFE == len(np.unique(idx))
        DES = RES[:-5] + ".npy"; np.save(open(DES + ".tmp", "wb"), idx); os.replace(DES + ".tmp", DES)
        json.dump(log, open(RES[:-5] + "_refits.json", "w"))
    tacq = time.time() - t0
    known = np.zeros(N, bool); known[idx] = True
    out = dict(arm=ARM, tag=TAG, seed=SEED, acq=ACQ, gate=GATE, labels=LABELS, feat=FEAT, qc=QC, batchrule=BR, batch=BATCH, design=DESIGN,
               nfe=int(len(idx)), t_acq_s=round(tacq, 1), design_failures=int(truth[idx].sum()),
               switch_on_nfe=t_on, ro_gate=RO_GATE)
    preds = readouts(idx, known)
    out.update({k: score(v, known) for k, v in preds.items()}); out["t_total_s"] = round(time.time() - t0, 1)
    np.savez_compressed(RES[:-5] + "_pred.npz", idx=idx, **preds)
    open(RES + ".tmp", "w").write(json.dumps(out, indent=1) + "\n"); os.replace(RES + ".tmp", RES)
    print(f"RESULT {os.path.basename(RES)[:-5]}: " + " | ".join(
        f"{k} D={v['D']:6.2f} FP={v['FP']:3d} FN={v['FN']:3d}" for k, v in out.items() if isinstance(v, dict))
        + f" | design failures {out['design_failures']} switch-on {t_on} [{out['t_total_s']:.0f}s]", flush=True)

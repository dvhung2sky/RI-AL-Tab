"""Example 4 VARIANT databases, rebuilt from the stored pushdowns with NO new FE runs.

Port of make_rbs_variant.py to the Xiao building. Ex4 as generated has no rupture mode at all (the
worst single-step load drop in 9,915 curves is 3.9 %), so `min(g1, g3)` is the exact limit state and
the structure-exploiting readout is complete by construction. This plants the missing mode.

THE MECHANISM, and why it is the right one for reinforced concrete. Ex3 uses an LEFM weld-flaw check,
which has no meaning here. The RC analogue at a beam-column interface is LAP-SPLICE / DEVELOPMENT-
LENGTH DEFICIENCY: the required development length goes as f_y d_b / sqrt(f_c), so with l_d the
length actually provided,

    r = f_y d_b / (sqrt(f_c) l_d),      b = 1[r > r*],      r* = the (1-p) quantile

A deficient splice cannot develop the bar, so the tie pulls out instead of yielding and the bridging
mechanism loses its tension tie at kappa times the rotation a sound splice would reach. d_b is a
design constant and cancels at the quantile threshold. f_y is U1 and f_c is U0 -- ALREADY population
inputs -- so the switch surface is oblique in (f_y, f_c, l_d), which an ARD kernel cannot align with,
exactly as in Ex3.

    theta_u^eff = kappa^b theta_u,0,     u_fr = theta_u^eff L_c      (chord rotation is kinematic:
                                                                      theta(u) = u / L_c exactly, so
                                                                      Ex3's stored rotation history
                                                                      is not needed here)

Past u_fr the tie is gone, so the curve is cut and P_ps decayed as P_ps(u_fr) u_fr / u, exactly as
fe_rbs.response does and make_rbs_variant.py reproduces.

TWO APPENDED RVs, columns 13 and 14, drawn from default_rng(1). Neither reaches OpenSees:
    13  theta_u,0  tie rotation capacity of a SOUND splice, mean THU_MED, CoV 0.25
    14  l_d        splice length as built, relative, CoV 0.20 (workmanship)

Knobs (env): FAD="p,kappa" (default 0.3,0.5), THU_MED, PF, THA_MED (read by fe_xiao at import).

    cd conformal-reliability/ex14fe && FAD=0.3,0.5 python3 make_xiao_variant.py
"""
import json
import os
import numpy as np
import fe_xiao as F

FAD = os.environ.get("FAD", "0.3,0.5")
PF = float(os.environ.get("PF", "1e-2"))
THU_MED = float(os.environ.get("THU_MED", "0.12"))     # [ASSUME] sound-splice tie rotation capacity,
COV_THU, COV_LD = 0.25, 0.20                           # rad. RC column-removal tests report bar
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ex4_variants")  # fracture at chord
os.makedirs(OUTD, exist_ok=True)                       # rotations of roughly 0.10-0.20 rad.

src = sorted(f for f in os.listdir(".") if f.startswith("resp_xiao_N") and f.endswith("_clean.npy"))[-1]
n = np.load(src, allow_pickle=True).item()
U, PD, UL, G = n["U"], n["Pd"].astype(float), n["ulast"], F.GRID
assert not n["bad"].any() and np.allclose(n["grid"], G)
N = len(U); rows = np.arange(N)

rng = np.random.default_rng(1)
u_thu, u_ld = rng.standard_normal(N), rng.standard_normal(N)
thu0 = F._ln(u_thu, COV_THU, THU_MED)
p, kap = map(float, FAD.split(","))
r = F._ln(U[:, 1], F._COV["fy"]) / (np.sqrt(F._ln(U[:, 0], F._COV["fc"])) * F._ln(u_ld, COV_LD))
rstar = np.quantile(r, 1 - p)
b = r > rstar
tu = np.where(b, kap * thu0, thu0)                     # effective tie rotation capacity
U = np.c_[U, u_thu, u_ld]

ufr = tu * F.L_CLEAR                                   # kinematic: theta(u) = u / L_c
ufr = np.where(ufr <= UL, ufr, np.inf)                 # never reached inside the window -> no rupture
fin = np.isfinite(ufr)


def at(M, x):
    """Row-wise linear interpolation of M (N x len(G)) at x."""
    j = np.clip(np.searchsorted(G, x), 1, len(G) - 1)
    w = np.clip((x - G[j - 1]) / (G[j] - G[j - 1]), 0.0, 1.0)
    return M[rows, j - 1] + w * (M[rows, j] - M[rows, j - 1])


Pdg = PD.copy()
PdU = np.where(fin, at(PD, np.where(fin, np.minimum(ufr, G[-1]), 0.0)), 0.0)
tail = fin[:, None] & (G[None, :] > ufr[:, None])
Pdg[tail] = (PdU[:, None] * np.where(fin, ufr, 0.0)[:, None] / np.maximum(G[None, :], 1e-9))[tail]
Pdmax = np.where(fin, np.maximum(np.where(tail, -np.inf, PD).max(1), PdU), PD.max(1))
ulast = np.where(fin, ufr, UL)

lo, hi = 1.0e2, 1.0e9
for _ in range(80):
    mid = 0.5 * (lo + hi)
    g, _ = F.margins(U, Pdg, Pdmax, ufr, ulast, mid)
    lo, hi = (mid, hi) if (g <= 0).mean() < PF else (lo, mid)
lam = 0.5 * (lo + hi)
g, mode = F.margins(U, Pdg, Pdmax, ufr, ulast, lam)
S0 = float(g.std()); f = g <= 0

# what the mode is WORTH: the failures that exist only because the splice is deficient
g0, _ = F.margins(U, PD, PD.max(1), np.where(thu0 * F.L_CLEAR <= UL, thu0 * F.L_CLEAR, np.inf),
                  np.where(thu0 * F.L_CLEAR <= UL, thu0 * F.L_CLEAR, UL), lam)
tag = f"c1_fad{FAD.replace(',', '-')}_thu{THU_MED:g}_t{F._THA_MED:g}_pf{PF:.0e}"
print(f"variant {tag}: N={N:,} NRV={U.shape[1]}  lambda={lam/1e3:.1f} kN  Pf={f.mean():.5f} ({int(f.sum())} failures)")
print(f"  modes of the failures: strength {int((mode[f] == 1).sum())} rupture {int((mode[f] == 2).sum())} "
      f"acceptance {int((mode[f] == 3).sum())}  | whole population "
      + " ".join(f"{k}:{int((mode == k).sum())}" for k in (1, 2, 3)))
print(f"  deficient splices {b.mean() * 100:.1f} % | tie ruptures inside the window {fin.mean() * 100:.1f} % "
      f"(deficient {fin[b].mean() * 100:.1f} %, sound {fin[~b].mean() * 100:.1f} %)")
print(f"  switch-induced failures (fail with the knockdown, safe without): {int((f & (g0 > 0)).sum())} "
      f"of {int(f.sum())} | failures that are deficient: {int((f & b).sum())}")

if not os.environ.get("CHECK"):
    rfn = os.path.join(OUTD, f"resp_xiao_{tag.rsplit('_t', 1)[0]}_N{N}.npy")
    cfn = os.path.join(OUTD, f"cache_xiao_{tag}_N{N}_seed0.npy")
    meta = dict(fad=FAD, thu_med=THU_MED, tha_med=F._THA_MED, pf=PF, source=src, Lc=F.L_CLEAR,
                extra_cols="13 = theta_u,0 (CoV 0.25), 14 = l_d (CoV 0.20), both default_rng(1)",
                criterion="r = f_y / (sqrt(f_c) l_d) > quantile(r, 1-p); theta_u^eff = kappa^b theta_u,0")
    np.save(rfn, dict(U=U, grid=G, Pd=Pdg.astype(np.float32), Pdmax=Pdmax, ufr=ufr, ulast=ulast,
                      bad=np.zeros(N, bool), meta=meta), allow_pickle=True)
    np.save(cfn, dict(U=U, m=g, S0=S0, lam=lam, nbad=0, mode=mode, theta_u=tu, b=b, logr=np.log(r / rstar),
                      meta=meta), allow_pickle=True)
    # POPF form, so ex3_cat.py / ak_mcs_cat.py run the ladder on Ex4 with no driver change at all
    # (snap_bench.py:86 convention: U, g normalised, B = flags N x NCONN, LOGR = signed switch distance).
    # The _L3 file swaps LOGR for the EFFECTIVE tie capacity, so FEAT=logr on it is the L3 rung.
    pz = os.path.join(OUTD, f"pop_xiao_{tag}_N{N}.npz")
    np.savez_compressed(pz, U=U, g=g / S0, B=b[:, None], LOGR=np.log(r / rstar)[:, None], lam=lam,
                        meta=json.dumps({k: v for k, v in meta.items()}))
    np.savez_compressed(pz.replace(".npz", "_L3.npz"), U=U, g=g / S0, B=b[:, None],
                        LOGR=np.log(tu)[:, None], lam=lam,
                        meta=json.dumps(dict(meta, note="LOGR = log(effective tie rotation capacity) = L3")))
    print(f"  wrote {cfn}\n  wrote {pz} (+ _L3)")

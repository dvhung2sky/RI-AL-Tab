"""Example 3 VARIANT databases, rebuilt from the no-stop responses with no new FE runs.

`resp_rbs_nostop_N49999.npy` stores the connection rotation th(u) on the full grid, so the fracture
rule `th >= theta_u` is a post-processing step: every curve below is a real OpenSees curve, cut at the
fracture point and decayed after it exactly as `fe_rbs.response` does (Pd(u_fr) * u_fr / u).

Knobs (env):
    NCONN    1 = the published model: ONE theta_u governs, i.e. both RBS hinges at the removed column
               are assumed to fracture at the same rotation (perfectly correlated coupons).
             2 = series system: the two hinges next to the removed column get INDEPENDENT theta_u
               (same LN(0.1615, 0.30)); the assembly fractures at the first, min(theta_1, theta_2).
               Consistent with the half model: by symmetry both hinges see the same th(u), and the
               model already ends at first fracture. theta_2 is a 14th RV appended as column 13,
               drawn from default_rng(1), so columns 0-12 are untouched.
    THA_MED  acceptance-rotation mean, read by fe_rbs at import (0.127 published; 0.209 = GSA 2003).
    PF       target failure probability (3e-3 published).

Output goes to ex14fe/ex3_variants/, NOT ex14fe/: ex3_phys.py picks `sorted(cache_rbs*)[-1]` from
ex14fe/, and a new file there would silently change which database it loads.

    cd conformal-reliability/ex14fe && NCONN=2 python3 make_rbs_variant.py
    NCONN=1 CHECK=1 python3 make_rbs_variant.py      # harness: must reproduce the published database
"""
import os
import numpy as np
import fe_rbs as F

NCONN = int(os.environ.get("NCONN", "1")); assert NCONN in (1, 2)
PF = float(os.environ.get("PF", "3e-3"))
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ex3_variants")
os.makedirs(OUTD, exist_ok=True)

n = np.load("resp_rbs_nostop_N49999.npy", allow_pickle=True).item()
U, TH, PD, UL = n["U"], n["th"].astype(float), n["Pd"].astype(float), n["ulast"]
assert not n["bad"].any()
N, G = len(U), F.GRID
rows = np.arange(N)

rng = np.random.default_rng(1)
tus, extra = [n["theta_u"]], []
if NCONN == 2:
    extra.append(rng.standard_normal(N))
    tus.append(F.A.RBS["theta_u"] * F._ln(extra[-1], F._COV["thu"]))
# FAD="p,kappa": LEFM brittle-weld check per connection, K_I ~ f_y sqrt(a) against K_Ic. Each connection
# gets its own flaw size a ~ LN(1, 0.5) and toughness K_Ic ~ LN(1, 0.3) (two appended RVs); the fraction
# p with the highest ratio is brittle and fractures at kappa * theta_u. The switch surface is OBLIQUE in
# (a, K_Ic, f_y), which an ARD kernel cannot align with. p and kappa are scenario values, not sourced
# (kappa = 0.5 matches TN 1669's WUF-B / RBS fracture-rotation ratio 0.081 / 0.1615).
FAD = os.environ.get("FAD")
if FAD:
    p, kap = map(float, FAD.split(","))
    fyr = F._ln(U[:, 0], F._COV["fy"])
    for j in range(len(tus)):
        ua, uk = rng.standard_normal(N), rng.standard_normal(N); extra += [ua, uk]
        r = fyr * np.sqrt(F._ln(ua, 0.5)) / F._ln(uk, 0.3)
        tus[j] = np.where(r > np.quantile(r, 1 - p), kap * tus[j], tus[j])
tu = np.minimum.reduce(tus)
if extra:
    U = np.c_[U, np.column_stack(extra)]


def at(M, x):
    """Row-wise linear interpolation of M (N x len(G)) at x (finite entries only)."""
    j = np.clip(np.searchsorted(G, x), 1, len(G) - 1)
    w = np.clip((x - G[j - 1]) / (G[j] - G[j - 1]), 0.0, 1.0)
    return M[rows, j - 1] + w * (M[rows, j] - M[rows, j - 1])


# fracture: first crossing of th(u) = theta_u, interpolated between grid points
hit = TH >= tu[:, None]
k = np.maximum(hit.argmax(1), 1)
t0, t1 = TH[rows, k - 1], TH[rows, k]
ufr_c = G[k - 1] + np.clip((tu - t0) / np.maximum(t1 - t0, 1e-12), 0, 1) * (G[k] - G[k - 1])
short = UL < F.U_MAX - 2.0 * F.DU                      # resistance-collapse guard fired (fe_rbs why=2)
frac = hit.any(1) & (ufr_c <= UL)
ufr = np.where(frac, ufr_c, np.where(short, UL, np.inf))
ufr = np.where(ufr >= F.U_FR_MIN, ufr, np.inf)         # fe_rbs tripwire; measured inactive
fin = np.isfinite(ufr)

Pdg = PD.copy()
PdU = np.where(fin, at(PD, np.where(fin, ufr, 0.0)), 0.0)
tail = fin[:, None] & (G[None, :] > ufr[:, None])
Pdg[tail] = (PdU[:, None] * np.where(fin, ufr, 0.0)[:, None] / np.maximum(G[None, :], 1e-9))[tail]
Pdmax = np.where(fin, np.maximum(np.where(tail, -np.inf, PD).max(1), PdU), PD.max(1))
ulast = np.where(fin, ufr, UL)

lo, hi = 1.0e2, 1.0e7
for _ in range(60):
    mid = 0.5 * (lo + hi)
    g, _ = F.margins(U, Pdg, Pdmax, ufr, ulast, mid)
    lo, hi = (mid, hi) if (g <= 0).mean() < PF else (lo, mid)
lam = 0.5 * (lo + hi)
g, mode = F.margins(U, Pdg, Pdmax, ufr, ulast, lam)
S0 = float(g.std()); f = g <= 0

tag = f"c{NCONN}" + (f"_fad{FAD.replace(',', '-')}" if FAD else "") + f"_t{F._THA_MED:g}_pf{PF:.0e}"
print(f"variant {tag}: N={N:,} NRV={U.shape[1]}  lambda={lam/1e3:.1f} kN  Pf={f.mean():.5f} ({int(f.sum())} failures)")
print(f"  modes strength {int((mode[f]==1).sum())} rupture {int((mode[f]==2).sum())} acceptance {int((mode[f]==3).sum())}"
      f"  | fractured in window {fin.mean()*100:.1f} %  median theta_u {np.median(tu):.4f}")

if os.environ.get("CHECK"):
    o = np.load("resp_rbs_N49999_clean.npy", allow_pickle=True).item()
    oc = np.load("cache_rbs_pf3e-03_N49999_seed0.npy", allow_pickle=True).item()
    of = oc["m"] <= 0
    both = fin & np.isfinite(o["ufr"])
    print(f"  CHECK vs published: fracture flag agrees {np.mean(fin == np.isfinite(o['ufr']))*100:.3f} % | "
          f"median |u_fr diff| {np.median(np.abs(ufr[both]-o['ufr'][both])):.2f} mm | "
          f"median |Pdmax rel diff| {np.median(np.abs(Pdmax-o['Pdmax'])/o['Pdmax'])*100:.3f} % | "
          f"lambda ratio {lam/oc['lam']:.5f} | failure sets: {int((f&of).sum())} shared, "
          f"{int((f&~of).sum())} new-only, {int((~f&of).sum())} published-only | "
          f"corr(g) {np.corrcoef(g/S0, oc['m']/oc['S0'])[0,1]:.5f}")
else:
    rfn = os.path.join(OUTD, f"resp_rbs_{tag.rsplit('_t', 1)[0]}_N{N}.npy")   # curves: NCONN, FAD only
    cfn = os.path.join(OUTD, f"cache_rbs_{tag}_N{N}_seed0.npy")
    meta = dict(nconn=NCONN, fad=FAD, tha_med=F._THA_MED, pf=PF, source="resp_rbs_nostop_N49999.npy",
                extra_cols="13.. from default_rng(1): [theta_2 if NCONN=2] then (a, K_Ic) per connection")
    np.save(rfn, dict(U=U, grid=G, Pd=Pdg.astype(np.float32), Pdmax=Pdmax, ufr=ufr, ulast=ulast,
                      bad=np.zeros(N, bool), meta=meta), allow_pickle=True)
    # theta_u: the EFFECTIVE fracture rotation (after series min and brittle switch), an analytic
    # function of the inputs; the phys2 readout in ex3_phys.py reads it instead of column 5.
    np.save(cfn, dict(U=U, m=g, S0=S0, lam=lam, nbad=0, mode=mode, theta_u=tu, meta=meta), allow_pickle=True)
    print(f"  wrote {cfn}")

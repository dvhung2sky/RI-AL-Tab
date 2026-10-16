"""Assemble the Ex4 population from the 10,000 stored Xiao-building pushdowns (Ex4/xiao_prod).

Mirrors fe_rbs.py's cache builder: curves -> P_ps on GRID -> bisect lambda to TARGET_PF -> cache.
Non-convergent runs are DROPPED, never imputed, and their tail concentration is reported per RV
(the intent's constraint, from Ex3's L_conn precedent).

    cd conformal-reliability/ex14fe && python3 build_xiao_pop.py
"""
import os
import sys
import time
import numpy as np
import fe_xiao as F

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Ex4", "xiao_prod")
NTOT = 10_000
t0 = time.time()
U = np.random.default_rng(0).standard_normal((NTOT, F.NRV))      # run_xiao.design("prod"), row i = sample i

Pdg = np.zeros((NTOT, len(F.GRID)), np.float32)
Pdmax = np.zeros(NTOT); ufr = np.zeros(NTOT); ulast = np.zeros(NTOT); bad = np.zeros(NTOT, bool)
for i in range(NTOT):
    d = np.load(os.path.join(SRC, f"{i:05d}.npy"), allow_pickle=True).item()
    assert d["i"] == i and np.allclose(d["u_row"], U[i]), f"row {i} does not match the design"
    r = d["res"]
    if r.get("reason") != "window":
        bad[i] = True; continue
    Pdg[i], Pdmax[i], ufr[i], ulast[i] = F.curve(r["u"], r["lf"], r["W"])
    if i % 2000 == 0:
        print(f"  {i:,}/{NTOT:,} [{time.time() - t0:.0f} s]", flush=True)

print(f"\ndropped {int(bad.sum())} non-convergent ({bad.mean() * 100:.2f} %). Tail concentration:")
for j, nm in enumerate(F.NAMES):
    t = U[bad, j].mean() * np.sqrt(bad.sum())
    lost = (bad & (U[:, j] < -2.0)).sum() / max((U[:, j] < -2.0).sum(), 1) * 100
    print(f"  {nm:12s} mean z {U[bad, j].mean():+6.3f}  t {t:+7.2f}  "
          f"| {lost:5.1f} % of the z < -2 tail lost" + ("   <-- CONCENTRATED" if abs(t) > 3 else ""))

ok = ~bad
U, Pdg, Pdmax, ufr, ulast = U[ok], Pdg[ok], Pdmax[ok], ufr[ok], ulast[ok]
N = len(U)
lo_, hi_ = 1.0e2, 1.0e9
for _ in range(80):                                              # fe_rbs.py:257-263
    mid = 0.5 * (lo_ + hi_)
    gg, _ = margins_ = F.margins(U, Pdg, Pdmax, ufr, ulast, mid)
    lo_, hi_ = (mid, hi_) if (gg <= 0).mean() < F.TARGET_PF else (lo_, mid)
lam = 0.5 * (lo_ + hi_)
g, mode = F.margins(U, Pdg, Pdmax, ufr, ulast, lam)
S0 = float(g.std())

np.save(f"resp_xiao_N{N}_clean.npy", dict(U=U, grid=F.GRID, Pd=Pdg, Pdmax=Pdmax, ufr=ufr,
                                          ulast=ulast, bad=np.zeros(N, bool)), allow_pickle=True)
fn = f"cache_xiao_pf{F.TARGET_PF:.0e}_N{N}_seed0.npy"
np.save(fn, dict(U=U, m=g, S0=S0, lam=lam, nbad=int(bad.sum()), mode=mode,
                 meta=dict(source="Ex4/xiao_prod", Lc=F.L_CLEAR, tha_med=F._THA_MED,
                           pf=F.TARGET_PF, ndrop=int(bad.sum()))), allow_pickle=True)
f = g <= 0
print(f"\n{fn}  [{(time.time() - t0) / 60:.1f} min]")
print(f"  lambda = {lam / 1e3:.1f} kN = {lam / (10_000 * 20.9):.2f} x the Test-2 pattern "
      f"(W = {2817.3:.0f} kN)   N = {N:,}   theta_allow mean = {F._THA_MED}   Lc = {F.L_CLEAR:.0f} mm")
print(f"  Pf = {f.mean():.5f} ({int(f.sum())} failures)  counting CoV = {1 / np.sqrt(max(f.sum(), 1)):.3f}")
print(f"  modes of the failures: " + "  ".join(
    f"{k} {nm} {int((mode[f] == k).sum())}" for k, nm in ((1, "strength"), (2, "rupture"), (3, "acceptance"))))
print(f"  modes of the whole population: " + "  ".join(
    f"{k} {int((mode == k).sum())}" for k in (1, 2, 3)))
print(f"  P_ps max: median {np.median(Pdmax) / 1e3:.0f} kN, CoV {Pdmax.std() / Pdmax.mean():.3f}")

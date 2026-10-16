"""Table 8: mean error counts by regime of the true state; regenerates brittle_split.txt. Read-only over run JSON."""
import numpy as np
from build_master import load
E3 = {"c1": "c1_fad0.3-0.25_t0.127_pf3e-03"}
V4 = "pop_xiao_c1_fad0.3-0.25_thu0.11_t0.085_pf1e-02_N9915"
K = ("FN_brittle", "FN_ductile", "FP_brittle", "FP_ductile")
def row(name, ds, key):
    r = [d[key] for d in ds if key in d]
    if not r: print(f"{name:24s} --"); return
    print(f"{name:24s} n={len(r):2d} " + " ".join(f"{k} {np.mean([x[k] for x in r]):5.2f}" for k in K)
          + f" sPf<0 in {sum(x['sPf'] < 0 for x in r)}/{len(r)}")
for v, t in E3.items():
    row(f"Ex3 {v} n300 A8 L0", load(f"ro_v_{t}_A8b2_s*_ladder.json"), "Tg+mem")
    row(f"Ex3 {v} n300 RI-AL-Tab", load(f"v_{t}_Moracle_s*.json", "Moracle"), "cat+mem oracle")
    for n in (100, 200):
        row(f"Ex3 {v} n{n} A8 L0", load(f"v_{t}_A8n{n}_s*.json", f"A8n{n}"), "Tg+mem")
        row(f"Ex3 {v} n{n} RI-AL-Tab", load(f"v_{t}_Moraclen{n}_s*.json", f"Moraclen{n}"), "cat+mem oracle")
row("Ex4 n100 A8 L0", load(f"v_{V4}_A8n100_s*.json", "A8n100"), "Tg+mem")
row("Ex4 n100 RI-AL-Tab", load(f"v_{V4}_M2orn100_s*.json", "M2orn100"), "cat+mem oracle")
row("Ex4 n100 AK L0", load(f"ak_{V4}_b100_L0_s*.json"), "AK")
row("Ex4 n100 AK L2", load(f"ak_{V4}_b100_L2_s*.json"), "AK")

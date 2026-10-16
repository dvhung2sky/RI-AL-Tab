"""Master results table for the A8-L2 paper, read-only over ex3cat_runs JSON. python3 build_master.py"""
import glob, json, os, numpy as np
CAT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ex2_rupture_margin", "ex3cat_runs")
def load(pat, arm=None):
    out = []
    for f in sorted(glob.glob(os.path.join(CAT, pat))):
        b = os.path.basename(f)
        if arm and b.split("_")[-2] != arm: continue
        if b.endswith(("_refits.json",)): continue
        out.append(json.load(open(f)))
    return out
def cell(ds, key):
    rows = [d[key] for d in ds if isinstance(d, dict) and key in d]
    if not rows: return None
    g = lambda k: np.array([r[k] for r in rows if k in r], float)
    t = np.array([d.get("t_total_s", np.nan) for d in ds if key in d], float)
    return dict(n=len(rows), D=g("D"), FP=g("FP"), FN=g("FN"), sPf=g("sPf"), t=t,
                seeds=[d.get("seed") for d in ds if key in d])
def fmt(c):
    if c is None: return "--"
    sd = np.std(c["D"], ddof=1) if c["n"] > 1 else 0
    tt = np.nanmean(c["t"]) if np.isfinite(c["t"]).any() else float("nan")
    return (f"D {c['D'].mean():6.2f}±{sd:5.2f} (n={c['n']:2d}) FP {c['FP'].mean():5.1f} FN {c['FN'].mean():5.1f} "
            f"sPf {c['sPf'].mean():+6.2f}% t {tt:6.0f}s seeds {c['seeds']}")
if __name__ == "__main__":
    E3 = [("Ex3 c1-0.25", "c1_fad0.3-0.25_t0.127_pf3e-03")]
    V4 = "pop_xiao_c1_fad0.3-0.25_thu0.11_t0.085_pf1e-02_N9915"
    print("D = 100(FP+FN)/n_fail ; sPf = signed relative Pf error in % (as stored) ; t = t_total_s mean\n")
    for name, tag in E3:
        print(f"== {name}  (n_fail = 150)")
        for n, suf in [(100, "_n100"), (200, "_n200"), (300, "")]:
            print(f"  n={n}")
            print("    AK-MCS L0         ", fmt(cell(load(f"ak_{tag}{suf}_L0_s*.json"), "AK")))
            print("    AK-MCS L2         ", fmt(cell(load(f"ak_{tag}{suf}_L2_s*.json"), "AK")))
            if n < 300:
                a = load(f"v_{tag}_A8n{n}_s*.json", f"A8n{n}")
                print("    AL-Tab L0 (A8)    ", fmt(cell(a, "Tg+mem")))
                print("    L2 readout-only   ", fmt(cell(a, "cat+mem oracle")))
                m = load(f"v_{tag}_Moraclen{n}_s*.json", f"Moraclen{n}")
                print("    RI-AL-Tab (Moraclen)", fmt(cell(m, "cat+mem oracle")))
            else:
                r = load(f"ro_v_{tag}_A8b2_s*_ladder.json")
                print("    AL-Tab L0 (A8b2)  ", fmt(cell(r, "Tg+mem")))
                print("    L2 readout-only   ", fmt(cell(r, "cat+mem oracle")))
                m = load(f"v_{tag}_Moracle_s*.json", "Moracle")
                print("    RI-AL-Tab (Moracle)", fmt(cell(m, "cat+mem oracle")))
                f = load(f"v_{tag}_final_s*.json", "final")
                print("    L1 method (final) ", fmt(cell(f, "cat+mem")))
    print(f"\n== Ex4 Xiao planted splice (n_fail = 100)")
    for n in (100, 150, 200, 300):
        s_ak = "" if n == 300 else f"_b{n}"
        a8arm = "A8" if n == 300 else f"A8n{n}"; l2arm = "M2or" if n == 300 else f"M2orn{n}"
        print(f"  n={n}")
        print("    AK-MCS L0         ", fmt(cell(load(f"ak_{V4}{s_ak}_L0_s*.json"), "AK")))
        print("    AK-MCS L2         ", fmt(cell(load(f"ak_{V4}{s_ak}_L2_s*.json"), "AK")))
        print("    AL-Tab L0 (A8)    ", fmt(cell(load(f"v_{V4}_{a8arm}_s*.json", a8arm), "Tg+mem")))
        print("    RI-AL-Tab (M2or)  ", fmt(cell(load(f"v_{V4}_{l2arm}_s*.json", l2arm), "cat+mem oracle")))

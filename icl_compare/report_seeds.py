"""Mean +- std over seeds 0-4 (campaign time: seeds 0-2) -> RESULTS_seeds.md. Campaign: Ex3 c1-0.25, CANDN=2000 (ex3cat_runs/v_*_ICL_<b>_c2000_s<seed>.json).
Fixed-design bench: seed s uses the TabICLv2 Moracle s design + its own 10k test draw (logs/bench_<b>[_s<seed>].json)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(os.path.dirname(HERE), "ex2_rupture_margin", "ex3cat_runs")
import sys
EX = sys.argv[1] if len(sys.argv) > 1 else "ex1"
T = {"ex1": "c1_fad0.3-0.25_t0.127_pf3e-03", "ex2": "pop_xiao_c1_fad0.3-0.25_thu0.11_t0.085_pf1e-02_N9915"}[EX]
BT = "" if EX == "ex1" else "_ex2"; SEEDS = (0, 1, 2, 3, 4)
M = [("TabICLv2 (ref)", "tabicl", ""), ("TabPFN-3", "tabpfn3", ""), ("TabPFN-3.5", "tabpfn35", ""), ("Mitra-v2", "mitra", ""),
     ("EXAONE-Tabular", "exaone", "²"), ("LimiX-2", "limix", ""), ("TabDPT", "tabdpt", ""), ("TabPFN-v2", "tabpfn2", ""),
     ("TabFM ¹", "tabfm", "²")]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def ms(v, fmt):
    v = [x for x in v if x is not None]
    if not v:
        return "n/a"
    return f"{np.mean(v):{fmt}} ± {np.std(v, ddof=1):{fmt}}" if len(v) > 1 else f"{v[0]:{fmt}}"


L = ["| Model | n | A8-L2 D | Fixed-design RMSE | 90% coverage | Interval score ↓ | Params | Peak memory (GB) | Campaign time (s) |",
     "|---|---|---|---|---|---|---|---|---|"]
for name, k, foot in M:
    R = [load(os.path.join(RUNS, f"v_{T}_ICL_{k}_c2000_s{s}.json")) for s in SEEDS]
    B = [load(os.path.join(HERE, "logs", f"bench_{k}{BT}.json" if s == 0 else f"bench_{k}{BT}_s{s}.json")) for s in SEEDS]
    TT = [r['t_total_s'] for r in R[:3] if r]   # seeds 3-4 timing contended; not reported
    R, B = [r for r in R if r], [b for b in B if b]
    mem = [b["peak_rss_GB"] + (b["mps_driver_GB"] or 0) for b in B]
    L.append(f"| {name} | {len(R)} | {ms([r['cat+mem oracle']['D'] for r in R], '.2f')} | "
             f"{ms([b['rmse_median'] for b in B], '.4f')} | {ms([b['cov90'] for b in B], '.3f')} | "
             f"{ms([b['interval_score'] for b in B], '.3f')} | {B[0]['params_M'] if B else 'n/a'}M | {ms(mem, '.1f')} | "
             f"{ms(TT, '.0f')}{foot} |")
L += ["", "Mean ± sample std over seeds 0-4 (n = campaigns completed); campaign time over seeds 0-2 only (seeds 3-4 contended). ¹ no distributional head: 1 member + QC=global. "
      "² MPS (Apple GPU); others CPU, 2 threads; memory for MPS models = RSS + GPU (upper bound, unified memory). "
      "Seeds 1-2 ran three lanes concurrently (LimiX | other CPU models | MPS), so times may be mildly inflated vs seed 0."]
open(os.path.join(HERE, f"RESULTS_seeds_{EX}.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

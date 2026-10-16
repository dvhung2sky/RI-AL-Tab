"""v3 figures WITHOUT Ex1 c2-0.5 and WITHOUT the per-regime Kriging arm (removed 2026-10-14; old figures in _with_perregime/): Fig 4 (budget), 5 (seeds), 6 (bias), 7 (split), 10 (ladder).
Copies of the logic in ../fig_budget.py, ../fig_results.py, ../fig_ladder.py with the c2 cells removed; the originals are untouched.
Reads stored run JSON only.  python3 fig_noc2.py   (writes PNG/PDF next to this file)"""
import os, sys, numpy as np
FIG = os.path.dirname(os.path.abspath(__file__)); OUT = FIG
os.chdir(FIG); sys.path.insert(0, FIG)
import fig_results as FR                       # needs cwd = FIG (it imports ../facts); __main__ guard keeps it from running
from _style import plt, BLUE, ORANGE, AQUA, YELLOW, G0, G2, GS, W
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, os.path.join(os.path.dirname(FIG), "facts")); from build_master import load, cell
os.chdir(OUT)
T1, V4 = FR.T1, FR.V4
METHODS = [m for m in FR.METHODS if "per regime" not in m[0]]          # per-regime Kriging dropped from the paper
def method_handles(): return [Line2D([], [], color=c, marker=m, ls=l, lw=1.6, ms=6, label=n) for n, c, m, l in METHODS]
def runs(pat, arm, key): return {d["seed"]: d[key] for d in load(pat, arm) if key in d}
def out(fig, name, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout(); fig.savefig(name + ".png", dpi=300); fig.savefig(name + ".pdf"); plt.close(fig)

SEEDS5 = range(5)                              # Fig. 5 shows seeds 0-4 only, so every method has the same five designs
def fig_seeds():
    cells = [("(a) Ex1 c1-0.25, $n$ = 300", T1, "", "ro_v_{t}_A8b2_s*_ladder.json", None, "v_{t}_Moracle_s*.json", "Moracle"),
             ("(b) Ex2, $n$ = 100", V4, "_b100", "v_{t}_A8n100_s*.json", "A8n100", "v_{t}_M2orn100_s*.json", "M2orn100")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.9)); rng = np.random.default_rng(0)
    labs = ["AK\nL0", "AK\nL2", "AL-Tab\nL0", "RI-AL-\nTab L2"]
    for ax, (ttl, t, suf, a8p, a8a, rip, ria) in zip(axs, cells):
        series = [(runs(f"ak_{t}{suf}_L0_s*.json", None, "AK"), G0, "o"), (runs(f"ak_{t}{suf}_L2_s*.json", None, "AK"), G2, "s"),
                  (runs(a8p.format(t=t), a8a, "Tg+mem"), BLUE, "o"),
                  (runs(rip.format(t=t), ria, "cat+mem oracle"), ORANGE, "D")]
        for i, (d, c, mk) in enumerate(series):
            s = sorted(k for k in d if k in SEEDS5); assert len(s) == 5, (ttl, i, sorted(d)); y = np.array([d[k]["D"] for k in s]); x = i + rng.uniform(-0.18, 0.18, len(y))
            held = np.array([5 <= k <= 9 for k in s]) if (i == 3 and t != V4) else np.zeros(len(s), bool)
            ax.scatter(x[~held], y[~held], s=22, color=c, marker=mk, alpha=0.85, linewidths=0, zorder=3)
            if held.any():
                ax.scatter(x[held], y[held], s=28, facecolors="white", edgecolors=c, marker=mk, linewidths=1.2, zorder=3, label="held-out seeds 5–9")
            ax.hlines(y.mean(), i - 0.3, i + 0.3, color=c, lw=2.2, zorder=4)
            print("seeds", ttl[:12], labs[i].replace("\n", " "), "mean %.2f n=%d" % (y.mean(), len(y)))
        ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=10.5); ax.set_title(ttl); ax.set_ylim(bottom=-1)
    axs[0].set_ylabel("$D$ per seed (%)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#6b6b6b", ms=6, label="one seed (one design)"), Line2D([], [], color="#6b6b6b", lw=2.2, label="mean over seeds 0–4")], loc="upper center", ncol=2, fontsize=11, frameon=False)
    out(fig, "fig_seeds", rect=(0, 0, 1, 0.92))

def fig_bias():
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.7))
    specs = [("(a) Ex1 c1-0.25", T1, [(100, "_n100"), (200, "_n200"), (300, "")], 8.2),
             ("(b) Ex2 planted lap splice", V4, [(100, "_b100"), (150, "_b150"), (200, "_b200"), (300, "")], 9.9)]
    for ax, (ttl, t, ns, cov) in zip(axs, specs):
        ex1 = t != V4
        def get(k, n, s):
            if k == 0: return cell(load(f"ak_{t}{s}_L0_s*.json"), "AK")
            if k == 1: return cell(load(f"ak_{t}{s}_L2_s*.json"), "AK")
            if k == 2:
                if ex1 and n == 300: return cell(load(f"ro_v_{t}_A8b2_s*_ladder.json"), "Tg+mem")
                arm = f"A8n{n}" if n < 300 else "A8"; return cell(load(f"v_{t}_{arm}_s*.json", arm), "Tg+mem")
            arm = ("Moracle" if ex1 else "M2or") + ("" if n == 300 else f"n{n}")
            return cell(load(f"v_{t}_{arm}_s*.json", arm), "cat+mem oracle")
        ax.axhline(0, color="#444444", lw=0.8)                    # MC CoV band removed 2026-10-15 (old figure in _withband/)
        for k, (lab, c, mk, ls) in enumerate(METHODS):
            xs, m, sd = [], [], []
            for n, s in ns:
                cc = get(k, n, s)
                if cc: xs.append(n); m.append(cc["sPf"].mean()); sd.append(np.std(cc["sPf"], ddof=1) if cc["n"] > 1 else 0)
            ax.errorbar(np.array(xs) + (k - 1.5) * 3, m, yerr=sd, color=c, marker=mk, ls=ls, ms=5, lw=1.5, capsize=2.5)
            print("bias", ttl[:14], lab, [round(float(v), 1) for v in m])
        ax.set_title(ttl); ax.set_xlabel("FE evaluations $n$"); ax.set_xticks([n for n, _ in ns])
    axs[0].set_ylabel(r"$\varepsilon_{P_f}$ (%)")
    fig.legend(handles=method_handles(), loc="lower center", ncol=4, fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
    out(fig, "fig_bias", rect=(0, 0.10, 1, 1))

def fig_split():
    K = [("FN_brittle", "missed,\nbrittle"), ("FN_ductile", "missed,\nductile"), ("FP_brittle", "false\nalarm,\nbrittle"), ("FP_ductile", "false\nalarm,\nductile")]
    C = [("(a) Ex1 c1-0.25, $n$ = 300", load(f"ro_v_{T1}_A8b2_s*_ladder.json"), load(f"v_{T1}_Moracle_s*.json", "Moracle")),
         ("(b) Ex2, $n$ = 100", load(f"v_{V4}_A8n100_s*.json", "A8n100"), load(f"v_{V4}_M2orn100_s*.json", "M2orn100"))]
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.6)); x = np.arange(4); w = 0.38
    for ax, (ttl, a, r) in zip(axs, C):
        ma = [np.mean([d["Tg+mem"][k] for d in a]) for k, _ in K]; mr = [np.mean([d["cat+mem oracle"][k] for d in r]) for k, _ in K]
        print("split", ttl[:14], [round(float(v), 2) for v in ma], [round(float(v), 2) for v in mr], len(a), len(r))
        for xs, vals, col in ((x - w / 2 - 0.01, ma, BLUE), (x + w / 2 + 0.01, mr, ORANGE)):
            for b in ax.bar(xs, vals, w, color=col):
                h = b.get_height(); ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.1f}" if h >= 0.05 else "0", ha="center", va="bottom", fontsize=9.5, color="#333333")
        ax.set_xticks(x); ax.set_xticklabels([k[1] for k in K], fontsize=10); ax.set_title(ttl); ax.set_ylim(0, max(ma + mr) * 1.15)
    axs[0].set_ylabel("mean count per design")
    fig.legend(handles=[Patch(color=BLUE, label="AL-Tab (L0)"), Patch(color=ORANGE, label="RI-AL-Tab (L2)")],
               loc="lower center", ncol=2, fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
    out(fig, "fig_split", rect=(0, 0.08, 1, 1))

def fig_budget():
    E3 = {"(a) Ex1 c1-0.25": "c1_fad0.3-0.25_t0.127_pf3e-03"}
    def pts(spec):
        xs, m, s = [], [], []
        for n, (pat, arm, key) in spec:
            c = cell(load(pat, arm), key)
            if c: xs.append(n); m.append(c["D"].mean()); s.append(np.std(c["D"], ddof=1) if c["n"] > 1 else 0)
        return np.array(xs), np.array(m), np.array(s)
    sty = {"AK-MCS (L0)": ("#a6a6a6", "o", "--"), "AK-MCS (L2)": ("#6b6b6b", "s", "-"),
           "AL-Tab (L0)": ("#2a78d6", "o", "--"), "RI-AL-Tab (L2)": ("#eb6834", "D", "-")}
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.7))
    for ax, (ttl, tag) in zip(axs, E3.items()):
        S = {"AK-MCS (L0)": [(n, (f"ak_{tag}{s}_L0_s*.json", None, "AK")) for n, s in [(100, "_n100"), (200, "_n200"), (300, "")]],
             "AK-MCS (L2)": [(n, (f"ak_{tag}{s}_L2_s*.json", None, "AK")) for n, s in [(100, "_n100"), (200, "_n200"), (300, "")]],
             "AL-Tab (L0)": [(100, (f"v_{tag}_A8n100_s*.json", "A8n100", "Tg+mem")), (200, (f"v_{tag}_A8n200_s*.json", "A8n200", "Tg+mem")), (300, (f"ro_v_{tag}_A8b2_s*_ladder.json", None, "Tg+mem"))],
             "RI-AL-Tab (L2)": [(100, (f"v_{tag}_Moraclen100_s*.json", "Moraclen100", "cat+mem oracle")), (200, (f"v_{tag}_Moraclen200_s*.json", "Moraclen200", "cat+mem oracle")), (300, (f"v_{tag}_Moracle_s*.json", "Moracle", "cat+mem oracle"))]}
        for k, sp in S.items():
            x, m, s = pts(sp); c, mk, ls = sty[k]; print("budget c1", k, [round(float(v), 2) for v in m])
            ax.errorbar(x, m, yerr=s, color=c, marker=mk, ls=ls, capsize=3, ms=5, lw=1.5, label=k)
        ax.set_title(ttl); ax.set_xlabel("FE evaluations $n$"); ax.set_xticks([100, 150, 200, 300])
    ax = axs[1]
    S = {"AK-MCS (L0)": [(n, (f"ak_{V4}{'' if n==300 else f'_b{n}'}_L0_s*.json", None, "AK")) for n in (100, 150, 200, 300)],
         "AK-MCS (L2)": [(n, (f"ak_{V4}{'' if n==300 else f'_b{n}'}_L2_s*.json", None, "AK")) for n in (100, 150, 200, 300)],
         "AL-Tab (L0)": [(n, (f"v_{V4}_{a}_s*.json", a, "Tg+mem")) for n, a in [(100, "A8n100"), (150, "A8n150"), (200, "A8n200"), (300, "A8")]],
         "RI-AL-Tab (L2)": [(n, (f"v_{V4}_{a}_s*.json", a, "cat+mem oracle")) for n, a in [(100, "M2orn100"), (150, "M2orn150"), (200, "M2orn200"), (300, "M2or")]]}
    for k, sp in S.items():
        x, m, s = pts(sp); c, mk, ls = sty[k]; print("budget ex2", k, [round(float(v), 2) for v in m])
        ax.errorbar(x, m, yerr=s, color=c, marker=mk, ls=ls, capsize=3, ms=5, lw=1.5, label=k)
    ax.set_title("(b) Ex2 planted lap splice"); ax.set_xlabel("FE evaluations $n$"); ax.set_xticks([100, 150, 200, 300])
    axs[0].set_ylabel("$D$ (%)")
    for a in axs: a.set_ylim(bottom=-2)
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, ls=ls, lw=1.6, ms=6, label=k) for k, (c, mk, ls) in sty.items()], loc="lower center", ncol=4, fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
    out(fig, "fig_budget", rect=(0, 0.10, 1, 1))

def fig_ladder():
    cols = ["Ex1 c1\nAK\n$n$=300", "Ex1 c1\nTabICL RO\n$n$=300", "Ex2\nAK\n$n$=100", "Ex2\nTabICL\n$n$=100"]
    g = {"R1": [0.483, 0.638, 0.336, 0.370], "L1": [0.837, 0.391, 0.503, 0.611], "L2": [1.000, 0.914, 1.000, 0.933], "L3": [0.993, 1.000, 0.866, 1.000]}
    x = np.arange(len(cols)); w = 0.2
    fig, ax = plt.subplots(figsize=(W, 3.8))
    for i, (k, c) in enumerate(zip(g, ["#eda100", "#1baf7a", "#eb6834", "#6b6b6b"])): ax.bar(x + (i - 1.5) * w, g[k], w, label=k, color=c)
    ax.set_xticks(x); ax.set_xticklabels(cols, fontsize=10.5); ax.set_ylabel(r"$\gamma$ (share of gap closed)")
    ax.axhline(1, color="k", lw=0.6, ls=":"); ax.set_ylim(0, 1.1); ax.legend(ncol=4, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    ax.grid(axis="x", visible=False); out(fig, "fig_ladder")

if __name__ == "__main__":
    for f in (fig_budget, fig_seeds, fig_bias, fig_split, fig_ladder): f(); print("ok", f.__name__)

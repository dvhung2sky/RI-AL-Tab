"""Section 4 result figures (8), read-only over the stored run JSON / prediction files.  python3 fig_results.py
Style: _style.py (Times New Roman 12 pt, print-width layouts). AL-Tab blue, RI-AL-Tab orange; Kriging = greys."""
import glob, json, os, sys, numpy as np
from _style import plt, save, BLUE, ORANGE, AQUA, YELLOW, G0, G2, GS, W
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, "../facts"); from build_master import load, cell, CAT
T1, T2 = "c1_fad0.3-0.25_t0.127_pf3e-03", "c2_fad0.3-0.5_t0.127_pf3e-03"
V4 = "pop_xiao_c1_fad0.3-0.25_thu0.11_t0.085_pf1e-02_N9915"
POP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ex14fe", "ex4_variants")
def runs(pat, arm, key): return {d["seed"]: d[key] for d in load(pat, arm) if key in d}
def legend_axis(ax, handles, **kw):
    ax.axis("off"); ax.legend(handles=handles, loc="center", **kw)
METHODS = [("AK-MCS L0", G0, "o", "--"), ("AK-MCS L2", G2, "s", "-"), ("AK-MCS L2, per regime", GS, "^", "-."),
           ("AL-Tab L0", BLUE, "o", "--"), ("RI-AL-Tab L2", ORANGE, "D", "-")]
def method_handles(): return [Line2D([], [], color=c, marker=m, ls=l, lw=1.6, ms=6, label=n) for n, c, m, l in METHODS]

# ---------------------------------------------------------------- 1. per-seed D
def fig_seeds():
    cells = [("(a) Ex1 c1-0.25, $n$ = 300", T1, "", "ro_v_{t}_A8b2_s*_ladder.json", None, "v_{t}_Moracle_s*.json", "Moracle"),
             ("(b) Ex1 c2-0.5, $n$ = 300", T2, "", "ro_v_{t}_A8b2_s*_ladder.json", None, "v_{t}_Moracle_s*.json", "Moracle"),
             ("(c) Ex2, $n$ = 100", V4, "_b100", "v_{t}_A8n100_s*.json", "A8n100", "v_{t}_M2orn100_s*.json", "M2orn100"),
             ("(d) Ex2, $n$ = 150", V4, "_b150", "v_{t}_A8n150_s*.json", "A8n150", "v_{t}_M2orn150_s*.json", "M2orn150")]
    fig, axs = plt.subplots(2, 2, figsize=(W, 6.6)); axs = axs.ravel(); rng = np.random.default_rng(0)
    labs = ["AK\nL0", "AK\nL2", "AK per\nregime", "AL-Tab\nL0", "RI-AL-\nTab L2"]
    for ax, (ttl, t, suf, a8p, a8a, rip, ria) in zip(axs, cells):
        series = [(runs(f"ak_{t}{suf}_L0_s*.json", None, "AK"), G0, "o"), (runs(f"ak_{t}{suf}_L2_s*.json", None, "AK"), G2, "s"),
                  (runs(f"ak_{t}{suf}_L2strat_s*.json", None, "AK"), GS, "^"), (runs(a8p.format(t=t), a8a, "Tg+mem"), BLUE, "o"),
                  (runs(rip.format(t=t), ria, "cat+mem oracle"), ORANGE, "D")]
        for i, (d, c, mk) in enumerate(series):
            s = sorted(d); y = np.array([d[k]["D"] for k in s]); x = i + rng.uniform(-0.18, 0.18, len(y))
            held = np.array([5 <= k <= 9 for k in s]) if (i == 4 and t != V4) else np.zeros(len(s), bool)
            ax.scatter(x[~held], y[~held], s=22, color=c, marker=mk, alpha=0.85, linewidths=0, zorder=3)
            if held.any():
                ax.scatter(x[held], y[held], s=28, facecolors="white", edgecolors=c, marker=mk, linewidths=1.2, zorder=3,
                           label="held-out seeds 5–9")
            ax.hlines(y.mean(), i - 0.3, i + 0.3, color=c, lw=2.2, zorder=4)
        ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=10.5); ax.set_title(ttl); ax.set_ylim(bottom=-1)
        if t == T1: ax.legend(fontsize=10, loc="upper right")
    for a in axs[::2]: a.set_ylabel("$D$ per seed (%)")
    save(fig, "fig_seeds")

# ---------------------------------------------------------------- 2. signed Pf error vs budget
def fig_bias():
    fig, axs = plt.subplots(2, 2, figsize=(W, 6.2)); axs = axs.ravel()
    specs = [("(a) Ex1 c1-0.25", T1, [(100, "_n100"), (200, "_n200"), (300, "")], 8.2),
             ("(b) Ex1 c2-0.5", T2, [(100, "_n100"), (200, "_n200"), (300, "")], 8.2),
             ("(c) Ex2 planted lap splice", V4, [(100, "_b100"), (150, "_b150"), (200, "_b200"), (300, "")], 9.9)]
    for ax, (ttl, t, ns, cov) in zip(axs, specs):
        ex1 = t != V4
        def get(k, n, s):
            if k == 0: return cell(load(f"ak_{t}{s}_L0_s*.json"), "AK")
            if k == 1: return cell(load(f"ak_{t}{s}_L2_s*.json"), "AK")
            if k == 2: return cell(load(f"ak_{t}{s}_L2strat_s*.json"), "AK")
            if k == 3:
                if ex1 and n == 300: return cell(load(f"ro_v_{t}_A8b2_s*_ladder.json"), "Tg+mem")
                arm = f"A8n{n}" if n < 300 else "A8"; return cell(load(f"v_{t}_{arm}_s*.json", arm), "Tg+mem")
            arm = ("Moracle" if ex1 else "M2or") + ("" if n == 300 else f"n{n}")
            return cell(load(f"v_{t}_{arm}_s*.json", arm), "cat+mem oracle")
        ax.axhspan(-cov, cov, color="#e8e8e8", zorder=0); ax.axhline(0, color="#444444", lw=0.8)
        ax.text(0.98, 0.04, f"MC CoV ±{cov}%", transform=ax.transAxes, ha="right", fontsize=10, color="#555555")
        for k, (lab, c, mk, ls) in enumerate(METHODS):
            xs, m, sd = [], [], []
            for n, s in ns:
                cc = get(k, n, s)
                if cc: xs.append(n); m.append(cc["sPf"].mean()); sd.append(np.std(cc["sPf"], ddof=1) if cc["n"] > 1 else 0)
            ax.errorbar(np.array(xs) + (k - 2) * 3, m, yerr=sd, color=c, marker=mk, ls=ls, ms=5, lw=1.5, capsize=2.5)
        ax.set_title(ttl); ax.set_xlabel("FE evaluations $n$"); ax.set_xticks([n for n, _ in ns])
    for a in axs[[0, 2]]: a.set_ylabel(r"$\varepsilon_{P_f}$ (%)")
    legend_axis(axs[3], method_handles() + [Patch(color="#e8e8e8", label="population MC CoV band")], fontsize=11)
    save(fig, "fig_bias")

# ---------------------------------------------------------------- 3. errors by regime
def fig_split():
    K = [("FN_brittle", "missed,\nbrittle"), ("FN_ductile", "missed,\nductile"), ("FP_brittle", "false\nalarm,\nbrittle"),
         ("FP_ductile", "false\nalarm,\nductile")]
    C = [("(a) Ex1 c1-0.25, $n$ = 300", load(f"ro_v_{T1}_A8b2_s*_ladder.json"), load(f"v_{T1}_Moracle_s*.json", "Moracle")),
         ("(b) Ex1 c2-0.5, $n$ = 300", load(f"ro_v_{T2}_A8b2_s*_ladder.json"), load(f"v_{T2}_Moracle_s*.json", "Moracle")),
         ("(c) Ex2, $n$ = 100", load(f"v_{V4}_A8n100_s*.json", "A8n100"), load(f"v_{V4}_M2orn100_s*.json", "M2orn100"))]
    fig, axs = plt.subplots(2, 2, figsize=(W, 6.2)); axs = axs.ravel(); x = np.arange(4); w = 0.38
    for ax, (ttl, a, r) in zip(axs, C):
        ma = [np.mean([d["Tg+mem"][k] for d in a]) for k, _ in K]; mr = [np.mean([d["cat+mem oracle"][k] for d in r]) for k, _ in K]
        for xs, vals, col in ((x - w / 2 - 0.01, ma, BLUE), (x + w / 2 + 0.01, mr, ORANGE)):
            bars = ax.bar(xs, vals, w, color=col)
            for b in bars:
                h = b.get_height(); ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.1f}" if h >= 0.05 else "0", ha="center",
                                            va="bottom", fontsize=9.5, color="#333333")
        ax.set_xticks(x); ax.set_xticklabels([k[1] for k in K], fontsize=10); ax.set_title(ttl)
        ax.set_ylim(0, max(ma + mr) * 1.15)
    for a in axs[[0, 2]]: a.set_ylabel("mean count per design")
    legend_axis(axs[3], [Patch(color=BLUE, label="AL-Tab (L0): 20 seeds on Ex1, 10 on Ex2"),
                         Patch(color=ORANGE, label="RI-AL-Tab (L2): 10 seeds")], fontsize=11)
    save(fig, "fig_split")

# ---------------------------------------------------------------- 4. ablation (horizontal bars)
def fig_ablation():
    cols = [("(a) Ex2, $n$ = 100", f"v_{V4}_A8n100_s*.json", "A8n100", f"v_{V4}_M2placn100_s*.json", "M2placn100",
             f"v_{V4}_M2orrepn100_s*.json", "M2orrepn100", f"v_{V4}_M2orn100_s*.json", "M2orn100"),
            ("(b) Ex2, $n$ = 150", f"v_{V4}_A8n150_s*.json", "A8n150", None, None,
             f"v_{V4}_M2orrepn150_s*.json", "M2orrepn150", f"v_{V4}_M2orn150_s*.json", "M2orn150"),
            ("(c) Ex1 c1-0.25, $n$ = 300", f"ro_v_{T1}_A8b2_s*_ladder.json", None, f"v_{T1}_placebo_s*.json", "placebo",
             f"v_{T1}_Moraclerep_s*.json", "Moraclerep", f"v_{T1}_Moracle_s*.json", "Moracle")]
    labs = ["AL-Tab (L0)", "placebo partition", "L2 readout only", "isotropic repulsion", "RI-AL-Tab (full)"]
    colors = [BLUE, G0, AQUA, YELLOW, ORANGE]
    fig, axs = plt.subplots(1, 3, figsize=(W, 3.6), sharey=True)
    for ax, (ttl, ap, aa, pp, pa, rp, ra, fp, fa) in zip(axs, cols):
        d = load(ap, aa)
        cs = [cell(d, "Tg+mem"), cell(load(pp, pa), "cat+mem") if pp else None, cell(d, "cat+mem oracle"),
              cell(load(rp, ra), "cat+mem oracle"), cell(load(fp, fa), "cat+mem oracle")]
        top = max(c["D"].mean() + (np.std(c["D"], ddof=1) if c["n"] > 1 else 0) for c in cs if c)
        for i, (c, col) in enumerate(zip(cs, colors)):
            y = 4 - i
            if c is None: ax.text(0.5, y, "not run", va="center", fontsize=10, color="#777777"); continue
            m = c["D"].mean(); s = np.std(c["D"], ddof=1) if c["n"] > 1 else 0
            ax.barh(y, m, 0.68, color=col, xerr=s, capsize=2.5, error_kw={"lw": 0.9, "ecolor": "#333333"})
            ax.text(m + s + top * 0.03, y, f"{m:.2f}", va="center", fontsize=10, color="#333333")
        ax.set_title(ttl, fontsize=11.5); ax.set_xlim(0, top * 1.35); ax.set_xlabel("$D$ (%)"); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(5)); axs[0].set_yticklabels(labs[::-1])
    save(fig, "fig_ablation")

# ---------------------------------------------------------------- 5. misspecified flags
def fig_misspec():
    true = np.load(f"{POP}/{V4}.npz")["B"][:, 0]
    rows = [("none", cell(load(f"v_{V4}_A8n150_s*.json", "A8n150"), "Tg+mem"), BLUE, None),
            ("exact", cell(load(f"v_{V4}_M2orn150_s*.json", "M2orn150"), "cat+mem oracle"), ORANGE, 1.0)]
    for k, lab, col in (("flip05", "5%\nflipped", AQUA), ("flip10", "10%\nflipped", AQUA),
                        ("q60", "threshold\n60th pct.", YELLOW), ("q80", "threshold\n80th pct.", YELLOW)):
        b = np.load(f"{POP}/misspec/pop_xiao_misspec_{k}_N9915.npz")["B"][:, 0]
        rows.append((lab, cell(load(f"v_pop_xiao_misspec_{k}_N9915_M2orn150_s*.json", "M2orn150"), "cat+mem oracle"), col, (b == true).mean()))
    fig, ax = plt.subplots(figsize=(W, 4.0))
    for i, (lab, c, col, ag) in enumerate(rows):
        m = c["D"].mean(); s = np.std(c["D"], ddof=1)
        ax.bar(i, m, 0.66, color=col, yerr=s, capsize=3, error_kw={"lw": 0.9, "ecolor": "#333333"})
        e = c["sPf"].mean(); txt = f"$\\varepsilon$ {'+' if e >= 0 else '−'}{abs(e):.1f}%" + (f"\nagree {ag:.3f}" if ag is not None else "")
        ax.text(i, m + s + 0.4, txt, ha="center", va="bottom", fontsize=10, color="#333333")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("$D$ (%), mean ± s.d. (5 seeds)"); ax.set_ylim(0, 27)
    ax.legend(handles=[Patch(color=BLUE, label="no regime information"), Patch(color=ORANGE, label="exact criterion"),
                       Patch(color=AQUA, label="random flag errors"), Patch(color=YELLOW, label="systematic threshold error")],
              fontsize=10, loc="upper left", ncol=2)
    save(fig, "fig_misspec")

# ---------------------------------------------------------------- 6. stopping rule
def fig_stop():
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.4))
    for s in range(5):
        tr = {}
        for f in glob.glob(os.path.join(CAT, f"ro_v_{V4}_M2or_s{s}_n*.json")):
            d = json.load(open(f)); tr[d["nfe"]] = d["cat+mem oracle"]
        tr[300] = json.load(open(os.path.join(CAT, f"v_{V4}_M2or_s{s}.json")))["cat+mem oracle"]
        ns = np.array(sorted(tr)); F = np.array([100 + tr[n]["FP"] - tr[n]["FN"] for n in ns]); D = np.array([tr[n]["D"] for n in ns])
        Fd = dict(zip(ns, F))
        stop = next(n for n in ns if n >= 80 and n - 20 in Fd and n - 40 in Fd
                    and abs(Fd[n] - Fd[n - 20]) <= 1 and abs(Fd[n - 20] - Fd[n - 40]) <= 1)
        for ax, y in ((axs[0], F), (axs[1], D)):
            ax.plot(ns, y, color=G2, lw=1.1, alpha=0.85, label="one seed" if s == 0 else None)
            ax.scatter([stop], [y[list(ns).index(stop)]], color=ORANGE, s=40, zorder=4, label="stopping point" if s == 0 else None)
    axs[0].axhline(100, color="#444444", lw=0.8, ls=":"); axs[0].text(295, 102, "$N_f$ = 100", ha="right", fontsize=10)
    axs[0].set_ylabel(r"predicted failures $\hat F(n)$"); axs[1].set_ylabel("$D$ (%)")
    axs[0].set_title("(a) predicted failure count"); axs[1].set_title("(b) set error")
    for a in axs: a.set_xlabel("FE evaluations $n$")
    axs[0].legend(fontsize=10, loc="lower right"); axs[1].legend(fontsize=10, loc="upper right")
    save(fig, "fig_stop")

# ---------------------------------------------------------------- 7. where the errors are (Ex2, n = 100, seed 0)
def fig_mechanism():
    P = np.load(f"{POP}/{V4}.npz"); g = P["g"]; L = P["LOGR"][:, 0]; truth = g <= 0; y = g / (1 + np.abs(g))
    fig, axs = plt.subplots(1, 2, figsize=(W, 4.1), sharey=True)
    for ax, (ttl, arm, key) in zip(axs, (("(a) AL-Tab (L0)", "A8n100", "Tg+mem"), ("(b) RI-AL-Tab (L2)", "M2orn100", "cat+mem oracle"))):
        z = np.load(os.path.join(CAT, f"v_{V4}_{arm}_s0_pred.npz")); pred = z[key]; idx = z["idx"]
        fn, fp = truth & ~pred, ~truth & pred
        ax.scatter(L[~truth], y[~truth], s=2, color="#cfcfcf", linewidths=0, rasterized=True)
        ax.scatter(L[truth & pred], y[truth & pred], s=9, color=GS, linewidths=0)
        ax.scatter(L[fn], y[fn], s=40, marker="X", color=ORANGE, linewidths=0)
        ax.scatter(L[fp], y[fp], s=30, marker="^", color=AQUA, linewidths=0)
        ax.scatter(L[idx], y[idx], s=16, facecolors="none", edgecolors=BLUE, linewidths=0.8)
        ax.axvline(0, color="#444444", lw=0.8, ls="--"); ax.axhline(0, color="#444444", lw=0.6)
        ax.set_title(f"{ttl}: {fn.sum()} missed, {fp.sum()} false alarms", fontsize=11)
        ax.set_xlabel(r"switch distance $\log(r/r^*)$")
        ax.text(0.03, 0.95, "sound", transform=ax.transAxes, fontsize=10, color="#444444", va="top")
        ax.text(0.97, 0.95, "deficient", transform=ax.transAxes, fontsize=10, color="#444444", va="top", ha="right")
    axs[0].set_ylabel(r"bounded margin $T(g)$")
    h = [Line2D([], [], marker="o", ls="", color="#bdbdbd", ms=4, label="safe"),
         Line2D([], [], marker="o", ls="", color=GS, ms=4, label="failure, found"),
         Line2D([], [], marker="X", ls="", color=ORANGE, ms=8, label="failure, missed"),
         Line2D([], [], marker="^", ls="", color=AQUA, ms=7, label="false alarm"),
         Line2D([], [], marker="o", ls="", mfc="none", mec=BLUE, ms=6, label="evaluated (100)")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("fig_mechanism.png", dpi=300); fig.savefig("fig_mechanism.pdf"); plt.close(fig)

# ---------------------------------------------------------------- 8. cost break-even
def fig_cost():
    c = np.logspace(-2, 3.2, 200)
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.7))
    cases = [("(a) Ex1 c1-0.25", 200, 747.1, 300, 446.6, 0.16), ("(b) Ex2", 150, 358.0, 200, 338.0, 527.0)]
    for ax, (ttl, nr, tr, na, ta, cfe) in zip(axs, cases):
        ax.loglog(c, (na * c + ta) / 3600, color=BLUE, lw=1.8, label=f"AL-Tab, {na} runs")
        ax.loglog(c, (nr * c + tr) / 3600, color=ORANGE, lw=1.8, label=f"RI-AL-Tab, {nr} runs")
        be = (tr - ta) / (na - nr); ax.axvline(be, color="#444444", lw=0.8, ls="--")
        ax.text(be * 1.2, 0.03, f"break-even\n{be:.1f} s", fontsize=10, transform=ax.get_xaxis_transform())
        ax.axvline(cfe, color=YELLOW, lw=1.6)
        ax.text(cfe * (1.2 if cfe < 100 else 0.8), 0.62 if cfe < 100 else 0.30, f"actual\n{cfe:g} s", fontsize=10,
                ha="left" if cfe < 100 else "right", transform=ax.get_xaxis_transform())
        ax.set_title(ttl); ax.set_xlabel("cost of one FE evaluation (s)"); ax.legend(fontsize=10, loc="upper left")
    axs[0].set_ylabel("total analysis time (h)")
    save(fig, "fig_cost")

if __name__ == "__main__":
    for f in (fig_seeds, fig_bias, fig_split, fig_ablation, fig_misspec, fig_stop, fig_mechanism, fig_cost):
        f(); print("ok", f.__name__)

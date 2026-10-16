"""Stopping rule on the five Ex1 c1-0.25 RI-AL-Tab designs (seeds 0-4): same procedure and rule as fig_stop() for Ex2 in
../fig_results.py. Readouts every 20 runs from the nested designs (ex2_rupture_margin/jobs_stop_ex1.txt), n = 300 from the run.
python3 fig_stop_ex1.py  -> fig_stop_ex1.png/.pdf here"""
import glob, json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _style import plt, ORANGE, G2, W
CAT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "ex2_rupture_margin", "ex3cat_runs")
T1, NF = "c1_fad0.3-0.25_t0.127_pf3e-03", 150

fig, axs = plt.subplots(1, 2, figsize=(W, 3.4))
for s in range(5):
    tr = {}
    for f in glob.glob(os.path.join(CAT, f"ro_v_{T1}_Moracle_s{s}_n*.json")):
        d = json.load(open(f)); tr[d["nfe"]] = d["cat+mem oracle"]
    tr[300] = json.load(open(os.path.join(CAT, f"v_{T1}_Moracle_s{s}.json")))["cat+mem oracle"]
    ns = np.array(sorted(tr)); F = np.array([NF + tr[n]["FP"] - tr[n]["FN"] for n in ns]); D = np.array([tr[n]["D"] for n in ns])
    Fd = dict(zip(ns, F))
    # same label-free rule as Ex2: change of at most one predicted failure over each of the last two 20-run steps
    stop = next((n for n in ns if n >= 80 and n - 20 in Fd and n - 40 in Fd
                 and abs(Fd[n] - Fd[n - 20]) <= 1 and abs(Fd[n - 20] - Fd[n - 40]) <= 1), None)
    print(f"seed {s}: n {list(ns)}\n  F {list(F)}\n  D {[round(float(x), 2) for x in D]}\n  stop {stop}"
          + (f" (D = {D[list(ns).index(stop)]:.2f})" if stop else " (rule never fires)"))
    for ax, y in ((axs[0], F), (axs[1], D)):
        ax.plot(ns, y, color=G2, lw=1.1, alpha=0.85, label="one seed" if s == 0 else None)
        if stop: ax.scatter([stop], [y[list(ns).index(stop)]], color=ORANGE, s=40, zorder=4, label="stopping point" if s == 0 else None)
axs[0].axhline(NF, color="#444444", lw=0.8, ls=":"); axs[0].text(298, NF + 9, f"$N_f$ = {NF}", ha="right", fontsize=10)
axs[0].set_ylabel(r"predicted failures $\hat F(n)$"); axs[1].set_ylabel("$D$ (%)")
axs[0].set_title("(a) predicted failure count"); axs[1].set_title("(b) set error")
for a in axs: a.set_xlabel("FE evaluations $n$")
axs[0].legend(fontsize=10, loc="lower right"); axs[1].legend(fontsize=10, loc="upper right")
fig.tight_layout(); out = os.path.join(HERE, "fig_stop_ex1"); fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".pdf"); plt.close(fig)

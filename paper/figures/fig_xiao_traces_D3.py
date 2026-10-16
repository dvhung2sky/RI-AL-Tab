"""D3-only version of figures/ex14fe_xiao2015_traces: measured D3 trace (Xiao et al. 2015, Fig. 6a, digitised) vs
model E. Test and model are zeroed at the pre-removal value and aligned at the onset of the drop, taken as the
first crossing of 20 % of the first extreme of each trace. Model history: run_xiao_D3_modelE.py."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "..", "..", "ex14fe", "xiao2015_traces_digitised.csv"), comment="#")
x = d[d.trace == "D3_II1"].sort_values("t_s")
tt, yt = x.t_s.values, x.disp_mm.values
tm, ym = np.load(os.path.join(HERE, "xiao_D3_modelE.npy")).T


# test: baseline = 0 mm before the drop; the first extreme is the -15.4 mm valley at 0.08 s after the drop
i_drop = np.argmax((tt > 0.3) & (yt < -1.0))                      # drop of the real trace (the first 0.1 s of the file is the plot frame)
peak1 = yt[i_drop:i_drop + 60].min()
t0_test = tt[i_drop + np.argmax(yt[i_drop:] <= 0.2 * peak1)]
t0_model = tm[np.argmax(ym <= 0.2 * ym[:int(0.2 / (tm[1] - tm[0]))].min())]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                     "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.4, 3.7), dpi=200)
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.plot(tt - t0_test, yt, color="#C0392B", alpha=0.6, lw=1.4, label="D3 test")
ax.plot(tm - t0_model, ym, color="#C0392B", lw=1.8, ls="-.", label="D3 model E")
ax.set(xlim=(-0.1, 0.7), ylim=(-16.5, 0.0), xlabel="time from onset of the drop (s)", ylabel="vertical displacement (mm)",
       title="side column D3 removed")
ax.grid(color="0.88", lw=0.6)
ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_xiao_traces_D3.png")); fig.savefig(os.path.join(HERE, "fig_xiao_traces_D3.pdf"))
print("t0 test %.3f raw s, t0 model %.3f s" % (t0_test, t0_model))

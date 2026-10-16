"""Component ablation (Table 9 / Fig. 9 type) at n = 100 and n = 300 on Ex1 c1-0.25 and Ex2; read-only over stored run JSON.
Writes fig_ablation_n100_n300.png/.pdf here and ../../ablation_n100_n300.docx.   python3 ablation_n100_n300.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "facts"))
from _style import plt, BLUE, ORANGE, AQUA, YELLOW, G0, W
from build_master import load, cell
import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

T1 = "c1_fad0.3-0.25_t0.127_pf3e-03"
V4 = "pop_xiao_c1_fad0.3-0.25_thu0.11_t0.085_pf1e-02_N9915"
ROWS = [("al", "AL-Tab (L0)", BLUE), ("ro", "AL-Tab (L1)\nflag-surrogate", AQUA), ("full", "RI-AL-Tab (full)", ORANGE)]   # isotropic + placebo dropped on request
# (title, AL-Tab design -> AL-Tab and flag-surrogate readout, isotropic file (unused since 2026-10-15), full)
# Moraclerepn100 (Ex1 n=100) and M2orrep (Ex2 n=300) were never run: those cells print "not run"
COLS = [("Ex1 c1-0.25, n = 100", (f"v_{T1}_A8n100_s*.json", "A8n100"), (f"v_{T1}_Moraclerepn100_s*.json", "Moraclerepn100"), (f"v_{T1}_Moraclen100_s*.json", "Moraclen100")),
        ("Ex1 c1-0.25, n = 300", (f"ro_v_{T1}_A8b2_s*_ladder.json", None), (f"v_{T1}_Moraclerep_s*.json", "Moraclerep"), (f"v_{T1}_Moracle_s*.json", "Moracle")),
        ("Ex2, n = 100", (f"v_{V4}_A8n100_s*.json", "A8n100"), (f"v_{V4}_M2orrepn100_s*.json", "M2orrepn100"), (f"v_{V4}_M2orn100_s*.json", "M2orn100")),
        ("Ex2, n = 300", (f"v_{V4}_A8_s*.json", "A8"), (f"v_{V4}_M2orrep_s*.json", "M2orrep"), (f"v_{V4}_M2or_s*.json", "M2or"))]

def cells(col):
    _, a, r, f = col
    d = load(*a)
    c = {"al": cell(d, "Tg+mem"), "ro": cell(d, "cat+mem oracle"), "full": cell(load(*f), "cat+mem oracle")}
    for k in ("al", "ro", "full"): assert c[k] is not None, (col[0], k)          # rep may still be running -> "not run"
    return c
def sd(c): return np.std(c["D"], ddof=1) if c["n"] > 1 else 0.0
def txt(c, seeds=False): return "not run" if c is None else f"{c['D'].mean():.2f} ± {sd(c):.2f}" + (f" ({c['n']})" if seeds else "")

R = [cells(col) for col in COLS]
for col, c in zip(COLS, R): print(col[0], {k: txt(v, True) for k, v in c.items()})

# ---------------- figure: 2 x 2 panels, same encoding as Fig. 9
fig, axs = plt.subplots(2, 2, figsize=(W, 4.6)); axs = axs.ravel()
for j, (ax, col, c) in enumerate(zip(axs, COLS, R)):
    top = max(5.0, max(v["D"].mean() + sd(v) for v in c.values() if v))   # floor: a near-zero panel is not blown up to fill the axis
    for i, (k, lab, colr) in enumerate(ROWS):
        y = len(ROWS) - 1 - i; v = c[k]
        if v is None: ax.text(top * 0.02, y, "not run", va="center", fontsize=10, color="#777777"); continue
        m, s = v["D"].mean(), sd(v)
        ax.barh(y, m, 0.68, color=colr, xerr=s, capsize=2.5, error_kw={"lw": 0.9, "ecolor": "#333333"})
        ax.text(m + s + top * 0.03, y, f"{m:.2f}", va="center", fontsize=10, color="#333333")
    ttl = col[0].replace("n = ", "$n$ = ")
    ax.set_title(f"({'abcd'[j]}) {ttl}", fontsize=11.5); ax.set_xlim(0, top * 1.35); ax.set_xlabel("$D$ (%)"); ax.grid(axis="y", visible=False)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[1] for r in ROWS][::-1] if j % 2 == 0 else [])
fig.tight_layout(); png = os.path.join(HERE, "fig_ablation_n100_n300"); fig.savefig(png + ".png", dpi=300); fig.savefig(png + ".pdf"); plt.close(fig)

# ---------------- docx
doc = docx.Document(); st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
doc.add_heading("Component ablation at n = 100 and n = 300", level=2)
p = doc.add_paragraph(); r = p.add_run("Table 9b. "); r.bold = True
p.add_run("Component ablation at n = 100 and n = 300: set error D (%), mean ± s.d. over seeds (seed counts in the notes below). "
          "AL-Tab (L1) flag-surrogate re-reads the AL-Tab (L0) designs with the regime flags as an extra surrogate input in the final readout only (no additional FE runs).")
t = doc.add_table(rows=1 + len(ROWS), cols=1 + len(COLS)); t.style = "Table Grid"
t.cell(0, 0).text = "Variant"
for j, col in enumerate(COLS): t.cell(0, j + 1).text = col[0]
labels = {"al": "AL-Tab (L0)", "ro": "AL-Tab (L1)\nflag-surrogate", "full": "RI-AL-Tab (full)"}
for i, (k, _, _) in enumerate(ROWS):
    t.cell(i + 1, 0).text = labels[k]
    for j, c in enumerate(R): t.cell(i + 1, j + 1).text = txt(c[k])
    if k == "full":
        for cc in t.rows[i + 1].cells:
            for run in cc.paragraphs[0].runs: run.bold = True
for row in t.rows:
    for cc in row.cells:
        for run in cc.paragraphs[0].runs: run.font.size = Pt(10.5)
doc.add_paragraph()
doc.add_picture(png + ".png", width=Inches(6.3)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph(); r = p.add_run("Fig. 9b. "); r.bold = True
p.add_run("Component ablation at n = 100 and n = 300 (values of Table 9b): D mean ± 1 s.d., with the mean beside each bar.")
miss = [f"{col[0]}: {labels[k]}" for col, c in zip(COLS, R) for k, _, _ in ROWS if c[k] is None]
thin = [f"{col[0]}: {labels[k]} ({c[k]['n']} seeds)" for col, c in zip(COLS, R) for k, _, _ in ROWS if c[k] and c[k]["n"] < 5]
doc.add_heading("Notes", level=3)
doc.add_paragraph("Seeds per cell: " + "; ".join(f"{col[0]}: " + ", ".join(f"{labels[k].replace(chr(10), ' ')} {c[k]['n']}" for k, _, _ in ROWS if c[k]) for col, c in zip(COLS, R)) + ".")
if miss: doc.add_paragraph("Cells without stored runs: " + "; ".join(miss) + ".")
if thin: doc.add_paragraph("Cells with fewer than five seeds: " + "; ".join(thin) + ".")
doc.add_paragraph("Source: stored run JSON in ex2_rupture_margin/ex3cat_runs (TabICL v2 runs, as in Tables 7–9 of v3_h); "
                  "generated by A8L2_paper/figures/v3/ablation_n100_n300.py.")
out = os.path.join(PAPER, "ablation_n100_n300.docx"); doc.save(out); print("saved", out)

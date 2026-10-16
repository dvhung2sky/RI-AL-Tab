# RI-AL-Tab: regime-informed active learning with an in-context tabular surrogate

Code, data and run records for the paper

> **Design checks as zero-cost information in active-learning reliability analysis of progressive collapse with brittle regimes**

The repository lets you (1) regenerate every results table and figure of Section 4 from the stored run records in
seconds, and (2) re-run any reported campaign (RI-AL-Tab, AL-Tab, AK-MCS) on the stored finite-element (FE)
databases. The scripts that generated the FE databases with OpenSees are included for reference.

## Setup

Tested with Python 3.14.6 on macOS (Apple silicon), CPU only.

```bash
pip install -r requirements.txt
python3 get_weights.py      # TabICL v2 checkpoints from Hugging Face (jingang/TabICL), checked by SHA-256
```

All commands below are run from the repository root (the code loads the checkpoints and the data by
root-relative paths).

## 1. Tables and figures from the stored records (seconds)

| Paper item | Command | Output |
|---|---|---|
| Table 5 (main comparison) | `cd paper/facts && python3 build_master.py` | stdout |
| Table 6 (error counts by regime) | `cd paper/facts && python3 brittle_split.py` | stdout |
| Table 7, Fig. 10 (component ablation) | `cd paper/figures && python3 ablation_n100_n300.py` | `fig_ablation_n100_n300.png`, stdout |
| Figs. 6–9 (budget, per-seed, signed error, error counts) | `cd paper/figures && python3 -c "import fig_noc2 as F; [f() for f in (F.fig_budget, F.fig_seeds, F.fig_bias, F.fig_split)]"` | `fig_budget/seeds/bias/split.png` |
| Fig. 11 (stopping rule, Ex1 / Ex2) | `cd paper/figures && python3 fig_stop_ex1.py && python3 -c "import fig_results as R; R.fig_stop()"` | `fig_stop_ex1.png`, `fig_stop.png` |
| Fig. 5 (Ex2 calibration trace) | `cd paper/figures && python3 fig_xiao_traces_D3.py` | `fig_xiao_traces_D3.png` |
| Table 9 (other in-context surrogates) | `cd icl_compare && python3 report_seeds.py ex1 && python3 report_seeds.py ex2` | `RESULTS_seeds_ex1/ex2.md` |

Every cell is the mean ± sample s.d. over seeds 0–4.

## 2. Re-running the campaigns

`jobs/` lists, one command per line, the exact configuration that produced every stored record (each line ends
with the record it writes). The configuration was read back from the records themselves.

| File | Content | Cost per line (1 CPU thread) |
|---|---|---|
| `jobs/campaign.txt` | 65 RI-AL-Tab and AL-Tab campaigns | 3–31 min |
| `jobs/akmcs.txt` | 70 AK-MCS campaigns (L0, L2) | 6 s – 4 min |
| `jobs/readout.txt` | 135 readouts on stored designs (AL-Tab n = 300 on Ex1, stopping-rule curves) | 16 s – 4 min |

A driver skips a run whose record already exists, so move the stored record away (or work in a copy) before
re-running it. Run one job at a time. A TabICL job used 2.9 GB peak memory with 2,000 candidates per refit (Table 9); the
default uses 6,000, and several jobs in parallel can exhaust a laptop.

```bash
# example: RI-AL-Tab, Ex2, 100 FE runs, seed 0
mv ex2_rupture_margin/ex3cat_runs/v_pop_xiao_c1_fad0.3-0.25_thu0.11_t0.085_pf1e-02_N9915_M2orn100_s0* /tmp/
grep "M2orn100_s0.json" jobs/campaign.txt | sh
```

### Names used in the code

| Paper | Code |
|---|---|
| Example 1 (steel RBS assembly, brittle-weld variant) | data `ex14fe/ex3_variants/cache_rbs_c1_fad0.3-0.25_t0.127_pf3e-03_N49999_seed0.npy`, tag `c1_fad0.3-0.25_t0.127_pf3e-03` |
| Example 2 (RC building, lap-splice variant) | data `ex14fe/ex4_variants/pop_xiao_c1_fad0.3-0.25_thu0.11_t0.085_pf1e-02_N9915.npz` |
| RI-AL-Tab | `ex3_cat.py` with `ACQ=cat GATE=oracle`; arms `Moracle*` (Ex1), `M2or*` (Ex2); result key `cat+mem oracle` |
| AL-Tab (L0) | `ex3_cat.py` with `ACQ=a8 GATE=const`; arms `A8*`; result key `Tg+mem` |
| AL-Tab (L1) flag-surrogate (Table 7) | result key `cat+mem oracle` of the AL-Tab records |
| AK-MCS L0 / L2 | `ak_mcs_cat.py` with `RUNG=L0` / `RUNG=L2`; result key `AK` |
| set error D, signed P_f error | record fields `D`, `sPf` |

Each record also stores `FP`, `FN`, the counts by regime, and the run time.

## Layout

```
ex2_rupture_margin/ex3_cat.py     RI-AL-Tab and AL-Tab driver (Algorithms 1-3)
ex2_rupture_margin/ak_mcs_cat.py  AK-MCS baseline with regime columns
ex2_rupture_margin/ex3cat_runs/   run records (JSON) and designs (.npy), seeds 0-4
ex2_rupture_margin/ex3_runs/      AL-Tab Ex1 n = 300 designs (seeds 0-4)
v1.py a5u_noise.py a9_tabicl.py ak_mcs.py   shared helpers (TabICL wrapper, initial design, Kriging kernel)
fe_*.py                           other benchmarks, imported by a5u_noise.py only; not used by this paper
ex14fe/fe_rbs.py nist_assembly.py Ex1 limit state and OpenSees model (NIST TN 1669 RBS assembly)
ex14fe/make_rbs_variant.py        Ex1 brittle-weld variant (LEFM check)
ex14fe/xiao2015_building.py fe_xiao.py build_xiao_pop.py make_xiao_variant.py   Ex2 OpenSees model, database, lap-splice variant
icl_compare/                      Table 9 report and its fixed-design benchmark logs
paper/                            table and figure scripts
jobs/                             one command per stored record
MANIFEST.sha256                   checksums of the distributed files
```

## What is not covered

- **The FE databases are not regenerated here.** The generation scripts are included, but Ex2 took about
  1,700 core-hours (527 s median per OpenSees run).
- **Table 9 campaigns for the other in-context models are not re-runnable from this repository.** TabPFN-3,
  EXAONE Tabular, LimiX-2, TabDPT and Mitra-v2 each ran in its own Python environment, with an adapter that is
  not included. The stored records and benchmark logs regenerate the table.
- **Table 8 (time per step) has no aggregation script here.** It was measured with `python3 -m cProfile` on
  `ex3_cat.py` (RI-AL-Tab, 100 FE runs, seed 0, one thread), and the script that grouped the profile into
  steps was not kept. The FE times per run (0.16 s for Ex1, 527 s for Ex2) come from the database generation.
- **Tables 1–4 and Figs. 1–4** are model descriptions and drawings, not computed results.
- **AL-Tab Ex1, n = 300, uses stored designs.** These designs came from an earlier copy of the driver (not
  included). `ex3_cat.py` with `ACQ=a8 GATE=const` implements the same acquisition. Adding
  `CHECK=<design.npy>` makes it assert, after every refit, that it selects the identical points. Verified for seed 0:
  the re-run selected the same 300 points as the stored design.

## Licence

Code: MIT (see `LICENSE`). Data and run records: CC BY 4.0. The TabICL checkpoints are distributed by their
authors under their own licence.

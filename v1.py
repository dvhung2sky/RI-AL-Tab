"""
Conformal-driven active learning for structural reliability — minimal v1.

Goal of this file: the smallest thing that actually demonstrates the idea, runs
on numpy + scikit-learn + scipy only (no UQpy / MAPIE install needed), and prints
real numbers you can trust.

It compares two UNCERTAINTY SOURCES driving the SAME active-learning reliability
loop, on two benchmarks:

    PROBLEM 1  four-branch (smooth)  — deterministic limit state. GP is near-optimal
                                       here, so this is the easy control case.
    PROBLEM 2  four-branch (noisy)   — every evaluation returns g(x) plus
                                       HETEROSCEDASTIC, HEAVY-TAILED noise (Student-t,
                                       larger out near the failure branches). This
                                       violates the GP's Gaussian / homoscedastic
                                       assumption, which is exactly where a
                                       distribution-free (conformal) uncertainty
                                       should earn its place. True Pf is still defined
                                       by the noise-free function, so accuracy is
                                       measurable.

The two arms:
    1. GP-variance : Gaussian-process posterior std            = the AK-MCS baseline
    2. Conformal   : distribution-free jackknife intervals on the SAME GP surrogate,
                     with a UCB-style acquisition that explores undersampled branches

Within a seed both arms share the population, the initial design, the acquisition
budget, and the SAME noisy labels for the same points (paired). The only difference
is the uncertainty source. Each arm is repeated over N_SEEDS independent worlds and
reported as mean +/- std, because single runs are noisy.

Run:  python3 v1.py
"""

import os
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DIM = 2
N_POP = 40_000     # Monte Carlo population size
N_SEEDS = 10       # independent repetitions
BUDGET = 50        # active-learning points added per arm (calls = N0 + BUDGET)
N0 = int(os.environ.get("N0", "12"))   # initial design size (env: hold N0+BUDGET fixed to sweep it)
ALPHA = 0.10       # conformal miscoverage (90% intervals)
JITTER = 1e-8      # numerical jitter only; the WhiteKernel below LEARNS the real noise

# Benchmarks. `noise` = heteroscedastic Student-t noise level on each evaluation.
# Both arms use a NOISE-AWARE GP (the WhiteKernel term in kernel() is fit by the
# optimizer), so the GP estimates the observation-noise level itself instead of
# being handed a fixed guess. This is the hardened, fair baseline.
PROBLEMS = [
    {"name": "four-branch (smooth)", "noise": 0.00, "tag": "smooth"},
    {"name": "four-branch (noisy)",  "noise": 0.15, "tag": "noisy"},
]


def kernel():
    # WhiteKernel term => the GP optimizer estimates the noise level from data.
    return (C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-1, 1e1))
            + WhiteKernel(1e-3, (1e-6, 1e1)))


# --------------------------------------------------------------------------- #
# Limit state: four-branch series system. Inputs x1, x2 ~ N(0, 1). Failure g<=0.
# The canonical multimodal benchmark (disjoint failure regions).
# --------------------------------------------------------------------------- #
def four_branch(X):
    x1, x2 = X[:, 0], X[:, 1]
    k = 6.0
    b1 = 3 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / np.sqrt(2)
    b2 = 3 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / np.sqrt(2)
    b3 = (x1 - x2) + k / np.sqrt(2)
    b4 = (x2 - x1) + k / np.sqrt(2)
    return np.minimum.reduce([b1, b2, b3, b4])


def make_world(seed, noise):
    """Population, the labels the arms observe (noisy if noise>0), the noise-free
    truth, and the reference Pf (always from the noise-free truth)."""
    rng = np.random.default_rng(seed)
    pop = rng.standard_normal((N_POP, DIM))
    g_true = four_branch(pop)
    if noise > 0:
        dof = 4
        z = rng.standard_t(dof, size=N_POP) / np.sqrt(dof / (dof - 2))   # unit-var, heavy tail
        scale = noise * (1.0 + 0.5 * np.linalg.norm(pop, axis=1) / np.sqrt(pop.shape[1]))  # het, dim-normalized
        y_obs = g_true + scale * z
    else:
        y_obs = g_true
    return pop, y_obs, g_true, float(np.mean(g_true <= 0))


def initial_doe(seed):
    """Same initial design for both arms within a seed (paired comparison)."""
    rng = np.random.default_rng(seed + 777)
    return list(rng.choice(N_POP, size=N0, replace=False))


def sign_ambiguity(mean, width):
    """Higher = the surrogate is more likely wrong about the sign of g here."""
    width = np.maximum(width, 1e-12)
    return np.exp(-0.5 * (mean / width) ** 2)


def final_pf(pop, idx, y):
    gp = GaussianProcessRegressor(kernel=kernel(), alpha=JITTER, normalize_y=True,
                                  n_restarts_optimizer=1, random_state=0)
    gp.fit(pop[idx], y[idx])
    return float(np.mean(gp.predict(pop) <= 0))


# --------------------------------------------------------------------------- #
# Arm 1: GP-variance (AK-MCS). Uncertainty = GP posterior std.
# --------------------------------------------------------------------------- #
def run_gp(pop, y, seed):
    idx = initial_doe(seed)
    for _ in range(BUDGET):
        gp = GaussianProcessRegressor(kernel=kernel(), alpha=JITTER, normalize_y=True,
                                      n_restarts_optimizer=1, random_state=0)
        gp.fit(pop[idx], y[idx])
        mu, sigma = gp.predict(pop, return_std=True)
        amb = sign_ambiguity(mu, sigma)
        amb[idx] = -1.0
        idx.append(int(np.argmax(amb)))
    return final_pf(pop, idx, y), idx


# --------------------------------------------------------------------------- #
# Arm 3 (control): GP mean + GP posterior std, but the SAME UCB acquisition as
# the conformal arm. It shares the uncertainty source with Arm 1 and the
# acquisition with Arm 2, so it isolates exploration from calibration. If it
# also survives the noise, the win was exploration, not conformal calibration.
# --------------------------------------------------------------------------- #
def run_gp_ucb(pop, y, seed):
    idx = initial_doe(seed)
    for _ in range(BUDGET):
        gp = GaussianProcessRegressor(kernel=kernel(), alpha=JITTER, normalize_y=True,
                                      n_restarts_optimizer=1, random_state=0)
        gp.fit(pop[idx], y[idx])
        mu, sigma = gp.predict(pop, return_std=True)
        amb = sigma * sign_ambiguity(mu, sigma)   # UCB exploration, GP uncertainty
        amb[idx] = -1.0
        idx.append(int(np.argmax(amb)))
    return final_pf(pop, idx, y), idx


# --------------------------------------------------------------------------- #
# Arm 2: Conformal on the SAME GP mean. Uncertainty = a distribution-free
# jackknife (leave-one-out) interval whose local scale is the distance to the
# nearest evaluated point. UCB-style acquisition (width * ambiguity) prefers the
# MOST uncertain near-boundary point, so undersampled branches keep pulling
# samples. Same surrogate mean as Arm 1; only the uncertainty source differs.
# --------------------------------------------------------------------------- #
def run_conformal(pop, y, seed):
    idx = initial_doe(seed)
    for _ in range(BUDGET):
        gp = GaussianProcessRegressor(kernel=kernel(), alpha=JITTER, normalize_y=True,
                                      n_restarts_optimizer=1, random_state=0)
        gp.fit(pop[idx], y[idx])
        mu = gp.predict(pop)
        fitted = gp.kernel_   # reuse the learned noise + length-scale in the LOO refits

        X, yy = pop[idx], y[idx]
        tree = cKDTree(X)
        knn_train = tree.query(X, k=2)[0][:, 1] + 1e-9   # 2nd neighbor (skip self)
        knn_pop = tree.query(pop, k=1)[0] + 1e-9

        n = len(X)
        loo = np.empty(n)
        for i in range(n):
            m = np.ones(n, bool); m[i] = False
            g_i = GaussianProcessRegressor(kernel=fitted, alpha=JITTER, normalize_y=True,
                                           optimizer=None)   # fixed hyperparams, fast
            g_i.fit(X[m], yy[m])
            loo[i] = abs(yy[i] - g_i.predict(X[i:i + 1])[0])
        q = np.quantile(loo / knn_train, 1 - ALPHA)
        width = q * knn_pop

        amb = width * sign_ambiguity(mu, width)   # explore the most uncertain boundary
        amb[idx] = -1.0
        idx.append(int(np.argmax(amb)))
    return final_pf(pop, idx, y), idx


def summarize(name, pfs, errs, calls):
    pfs, errs = np.array(pfs), np.array(errs)
    print(f"  {name:<13} Pf = {pfs.mean():.2e} +/- {pfs.std():.0e}   "
          f"rel.err = {errs.mean():5.1f}% +/- {errs.std():4.1f}%   calls = {calls}")


def plot_samples(seed, noise, panels, fname, suptitle):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (plot skipped: matplotlib unavailable — {e})")
        return

    pop, _, g_true, pf_ref = make_world(seed, noise)
    lin = np.linspace(-5, 5, 400)
    xx, yy = np.meshgrid(lin, lin)
    gg = four_branch(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (title, idx, pf) in zip(axes, panels):
        ax.contourf(xx, yy, gg, levels=[gg.min(), 0.0], colors=["#f6c6c6"])
        ax.contour(xx, yy, gg, levels=[0.0], colors="k", linewidths=1.0)
        init, added = pop[idx[:N0]], pop[idx[N0:]]
        ax.scatter(init[:, 0], init[:, 1], s=24, c="#7f7f7f", marker="s",
                   label=f"initial ({N0})", zorder=3)
        ax.scatter(added[:, 0], added[:, 1], s=20, c="#1f77b4",
                   label=f"active ({len(added)})", zorder=3)
        err = abs(pf - pf_ref) / pf_ref * 100
        ax.set_title(f"{title}\nPf = {pf:.2e}  (true {pf_ref:.2e}, err {err:.0f}%)",
                     fontsize=10)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_xlabel("x1")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    axes[0].set_ylabel("x2")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"  plot saved: {fname}")


def run_problem(prob):
    print(f"\n=== {prob['name']}  (noise={prob['noise']}) "
          f"==============================")
    runners = [("GP/U-func", run_gp), ("GP/UCB", run_gp_ucb), ("Conformal", run_conformal)]
    refs = []
    pf = {n: [] for n, _ in runners}
    err = {n: [] for n, _ in runners}
    idx0 = {n: None for n, _ in runners}
    for s in range(N_SEEDS):
        pop, y, g_true, pf_ref = make_world(s, prob["noise"])
        refs.append(pf_ref)
        parts = [f"  seed {s}:  ref={pf_ref:.2e}"]
        for name, fn in runners:
            p, idx = fn(pop, y, s)
            pf[name].append(p); err[name].append(abs(p - pf_ref) / pf_ref * 100)
            if s == 0: idx0[name] = idx
            parts.append(f"{name}={p:.2e}")
        print("   ".join(parts))

    refs = np.array(refs)
    print(f"  reference Pf = {refs.mean():.2e} +/- {refs.std():.0e}")
    calls = N0 + BUDGET
    for name, _ in runners:
        summarize(name, pf[name], err[name], calls)
    panels = [(name, idx0[name], pf[name][0]) for name, _ in runners]
    plot_samples(0, prob["noise"], panels, f"samples_{prob['tag']}.png",
                 f"{prob['name']}: where each method sampled (pink = failure g<=0)")
    return {name: (float(np.mean(err[name])), float(np.std(err[name]))) for name, _ in runners}


if __name__ == "__main__":
    print(f"N_pop = {N_POP:,}  |  seeds = {N_SEEDS}  |  budget = {N0 + BUDGET} calls/arm")
    t0 = time.time()
    rows = []
    for prob in PROBLEMS:
        rows.append((prob["name"], run_problem(prob)))

    arms = ["GP/U-func", "GP/UCB", "Conformal"]
    print("\n========== SUMMARY (mean +/- std rel. error in Pf) ==========")
    print("  " + f"{'problem':<22}" + "".join(f"{a:<18}" for a in arms))
    for name, d in rows:
        cells = "".join(f"{d[a][0]:4.1f}% +/- {d[a][1]:4.1f}%  " for a in arms)
        print(f"  {name:<22}{cells}")
    print(f"\nelapsed {time.time()-t0:.0f}s")

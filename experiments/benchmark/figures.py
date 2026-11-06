"""Figures of the extended article, from the collected CSVs in results/.

    python -m experiments.benchmark.figures             # every figure
    python -m experiments.benchmark.figures noise       # one of them

Writes a PDF and a PNG per figure into experiments/benchmark/figures/. The privacy figure
reads the CSVs that experiments/deployment/dp_*.py and cohorts.py write.

Two rules keep these readable in print: at most seven series in a panel, and a
series is identified by hue, marker and dash together, never by hue alone.
"""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures")
DEPLOYMENT = os.path.join(os.path.dirname(HERE), "deployment", "results")

# Draw the canvas large and let LaTeX shrink it by SHRINK. The elsarticle preprint
# layout is one 390 pt (5.4 in) column; single-column figures go in at 0.6 of it.
SHRINK = 0.59
SINGLE, DOUBLE = 0.6 * 5.4 / SHRINK, 5.4 / SHRINK


PANEL = "#eaeaf2"
# Notes drawn inside a panel; one step under the tick labels, on the canvas scale.
NOTE = 10.5
# One hue per algorithm; the reference sorts are grey, since they are not methods.
STYLE = {
    "FedRCD-X":     dict(color="#27518a", marker="o", ls="-"),
    "FedRCD-NI":    dict(color="#a65f2e", marker="s", ls="-"),
    "FedISHC":      dict(color="#188768", marker="^", ls="-"),
    "NOTEARS-ADMM": dict(color="#7a4a9e", marker="D", ls="-"),
    "DirectLiNGAM": dict(color="#922856", marker="v", ls="-"),
    "FedHC":        dict(color="#128aab", marker="P", ls="-"),
    "FedRCD-Def":   dict(color="#6772b6", marker="X", ls="-"),
    "ICA-LiNGAM":   dict(color="#d46262", marker="<", ls="-"),
    "VarSort":      dict(color="#3f3e3b", marker=None, ls="--"),
    "R2Sort":       dict(color="#6b6a65", marker=None, ls="-."),
    "Random":       dict(color="#85847f", marker=None, ls=":"),
}
# The same methods in every results figure and table: the family, the federated
# baselines, the centralised references and the reference sorts. HC and HC-LiNGAM
# return the orders of FedRCD-X and FedRCD-NI on every draw, so they are omitted.
ROSTER = ["FedRCD-NI", "FedRCD-Def", "FedRCD-X", "FedHC", "FedISHC", "NOTEARS-ADMM",
               "DirectLiNGAM", "ICA-LiNGAM", "VarSort", "R2Sort", "Random"]
FEDERATED = ROSTER[:6]
MAIN = ROSTER[:8]
NULLS = ["VarSort", "R2Sort", "Random"]
SETTING_LABEL = {"raw": "Raw", "std": "Standardised"}
INK, INK2 = "#0b0b0b", "#52514e"
BAND, CONNECT = "#d7d7e4", "#c3c3d3"
SYMMETRIC = ["logistic", "uniform", "laplace", "t"]
CHANCE = dict(color=INK, ls=(0, (1, 1.6)), lw=1.5)




def chance(ax, level=0.5):
    ax.axhline(level, zorder=1.5, **CHANCE)


def chance_handle():
    from matplotlib.lines import Line2D
    return Line2D([], [], label="Chance", **CHANCE)


DIST_LABEL = {"gaussian_cubed": "Gauss$^3$", "exponential_cubed": "Exp$^3$", "t": "Student-$t$"}


def setup():
    """Seaborn's darkgrid, in the house style, tuned for print.

    The panel carries a light fill with white gridlines and no spines. Seaborn's own
    darkgrid fill is a touch heavy once it is inked across eleven figures, so the
    gridlines are thinner here and the palette was re-checked against this panel
    rather than against white.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": PANEL, "axes.edgecolor": PANEL,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.4,
        "grid.linestyle": "-",
        "xtick.bottom": False, "ytick.left": False,
        "axes.axisbelow": True, "lines.linewidth": 2.4, "lines.markersize": 7,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def load(*exps):
    d = pd.concat([pd.read_csv(os.path.join(RESULTS, f"{e}.csv"), low_memory=False) for e in exps],
                  ignore_index=True)
    return d[d["error"].fillna("").astype(str).str.len() == 0]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)



def lines(ax, d, x, methods, val="anc_acc", logx=False, band=True, categorical=None,
          cat_labels=None, rotate=35):
    """One line per method. `categorical` gives the x order for an evenly spaced axis."""
    if "setting" in d and len(d) and (d.setting == "std").all():
        methods = [m for m in methods if m != "VarSort"]       # every variance is one
    for m in methods:
        sub = d[d.method == m]
        if sub.empty:
            continue
        g = sub.groupby(x)[val]
        mu, se = g.mean(), g.sem()
        if categorical is not None:
            mu, se = mu.reindex(categorical), se.reindex(categorical)
            xs = np.arange(len(categorical))
        else:
            xs = mu.index.values
        st = STYLE[m]
        ax.plot(xs, mu.values, color=st["color"], ls=st["ls"], marker=st["marker"], label=m,
                lw=1.7 if m in MAIN else 1.2, zorder=3 if m in MAIN else 2,
                markeredgecolor="white", markeredgewidth=0.6)
        if band:
            ax.fill_between(xs, mu - se, mu + se, color=st["color"], alpha=0.13, lw=0, zorder=1)
    if logx:
        ax.set_xscale("log")
    if categorical is not None:
        ax.set_xticks(np.arange(len(categorical)))
        labels = cat_labels if cat_labels is not None else [
            DIST_LABEL.get(t, t.replace("_", " ").capitalize()) for t in categorical]
        # rotation_mode="anchor" pivots the label about its alignment point instead of
        # its centre. Without it matplotlib rotates first and aligns the bounding box
        # after, so a long name drifts away from the tick it belongs to.
        ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center",
                           rotation_mode="anchor" if rotate else None)
        ax.grid(axis="x", visible=False)


def legend_order():
    return ROSTER + MAIN + NULLS



def legend_right(fig, axes, width=0.78, y=0.5, bottom=0.0):
    """Legend as a column beside the panels instead of a strip beneath them.

    A strip under the axes eats height that the plot needs, and it does so unevenly:
    figures that carry one lose plot area that figures without one keep, so a single
    height budget applied to both squeezes the wrong ones. Put beside the panels, the
    legend costs width, which these figures have, and the axes keep the full height.
    """
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    order = [l for l in dict.fromkeys(legend_order()) if l in labels]
    fig.tight_layout(rect=(0, bottom, width, 1))
    fig.legend([handles[labels.index(l)] for l in order], order, loc="center left",
               bbox_to_anchor=(width, y), ncol=1, handlelength=2.2,
               borderaxespad=0.0, labelspacing=0.5)



def main_cell(*exps):
    d = load(*exps)
    return d[(d.p == 20) & (d.K == 10)]



def fig_instrument():
    """Three scores from the same aggregated cumulants, raw data: two read scale, one does not.

    tau4 (FedRCD-NI) and the residual score M4 carry the marginal scales; M4n is M4
    divided by sigma_i^3 sigma_j, the same criterion with the variance ladder removed.
    """
    scores = [("FedRCD-NI", r"$\tau^{(4)}$ (FedRCD-NI)", dict(color="#a65f2e", marker="s", ls="-")),
              ("FedRCD-NI-M", r"$M^{(4)}$, with scale", dict(color="#27518a", marker="o", ls="-")),
              ("FedRCD-NI-Mn", r"$M^{(4\mathrm{n})}$, scale-free", dict(color="#922856", marker="v", ls="-")),
              ("VarSort", "VarSort", STYLE["VarSort"]),
              ("R2Sort", "R2Sort", STYLE["R2Sort"])]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.7), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.35]})
    d = load("E_sample", "E_sample2")
    d = d[(d.setting == "raw") & (d.dist != "mixed")]
    m = main_cell("P_main")
    m = m[m.setting == "raw"]
    fams = [f for f in SYMMETRIC if f in set(m.dist)] + \
           [f for f in ["gaussian_cubed", "poisson", "exponential", "gamma", "pareto",
                        "exponential_cubed", "mixed"] if f in set(m.dist)]
    for ax, data, x in ((axes[0], d, "n"), (axes[1], m, "dist")):
        for meth, label, st in scores:
            g = data[data.method == meth].groupby(x).anc_acc
            mu, se = g.mean(), g.sem()
            if x == "dist":
                mu, se = mu.reindex(fams), se.reindex(fams)
                xs = np.arange(len(fams))
            else:
                xs = mu.index.values
            ax.plot(xs, mu.values, label=label, lw=1.7, markeredgecolor="white", markeredgewidth=0.6, **st)
            ax.fill_between(xs, mu - se, mu + se, color=st["color"], alpha=0.13, lw=0)
        chance(ax)
        ax.set_ylim(0.32, 1.02)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Total sample size $N$")
    axes[0].set_ylabel(r"AncAcc $\uparrow$")
    axes[0].set_title("Against the sample size")
    axes[1].set_xticks(np.arange(len(fams)))
    axes[1].set_xticklabels([DIST_LABEL.get(f, f.replace("_", " ").capitalize()) for f in fams],
                            rotation=35, ha="right", rotation_mode="anchor")
    axes[1].grid(axis="x", visible=False)
    axes[1].set_title(r"Per noise family, $N = 2000$")
    from matplotlib.patches import Patch
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.legend(h + [chance_handle()], lab + ["Chance"], loc="center left", bbox_to_anchor=(0.8, 0.5),
               handlelength=2.2, borderaxespad=0.0, labelspacing=0.6)
    save(fig, "fig_instrument")



def fig_violations():
    """One assumption broken at a time, as a dot plot rather than a line tangle."""
    d = load("E_violations")
    d = d[d.setting == "raw"]
    kinds = [("clean", "Clean"), ("confounded", "Latent confounder"),
             ("measurement", "Measurement noise"), ("nonlinear_tanh", "Tanh edges"),
             ("nonlinear_square", "Square edges"), ("nonlinear_sin", "Sine edges"),
             ("gaussian", "Gaussian noise")]
    methods = ROSTER
    fig, ax = plt.subplots(figsize=(DOUBLE, 3.95))
    y = np.arange(len(kinds))[::-1]
    off = np.linspace(-0.36, 0.36, len(methods))
    order = [a for a, _ in kinds]
    for k, m in enumerate(methods):
        g = d[d.method == m].groupby("violation").anc_acc
        mu, se = g.mean().reindex(order), g.sem().reindex(order)
        st = STYLE[m]
        ax.errorbar(mu.values, y + off[k], xerr=se.values, fmt="none", ecolor=st["color"],
                    elinewidth=1.0, capsize=1.8, alpha=0.75, zorder=2)
        ax.scatter(mu.values, y + off[k], s=46, marker=st["marker"] or "o", label=m, zorder=3,
                   facecolor=st["color"], edgecolor="white", linewidth=0.7)
    for yi in y:
        ax.axhline(yi, color=CONNECT, lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([t for _, t in kinds])
    ax.set_xlabel(r"AncAcc $\uparrow$")
    ax.set_xlim(0.4, 1.03)
    ax.grid(axis="y", visible=False)
    legend_right(fig, ax, width=0.825)
    save(fig, "fig_violations")



def _no_random(methods):
    return [m for m in methods if m != "Random"]


def fig_noise():
    """Per noise family, raw data: AncAcc above, SHD below, legend beside both."""
    d = main_cell("P_main")
    d = d[d.setting == "raw"]
    fams = [f for f in SYMMETRIC if f in set(d.dist)] + \
           [f for f in ["gaussian_cubed", "poisson", "exponential", "gamma", "pareto",
                        "exponential_cubed", "mixed"] if f in set(d.dist)]
    methods = _no_random(ROSTER)
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 5.6), sharex=True)
    lines(axes[0], d, "dist", methods, categorical=fams)
    lines(axes[1], d, "dist", methods, val="lfix_shd", categorical=fams)
    axes[0].set_ylabel(r"AncAcc $\uparrow$")
    axes[1].set_ylabel(r"SHD $\downarrow$")
    axes[0].tick_params(labelbottom=False)
    for ax in axes:
        ax.axvspan(-0.5, len(SYMMETRIC) - 0.5, color=BAND, zorder=0)
        ax.axvline(len(SYMMETRIC) - 0.5, color=INK2, lw=0.8, alpha=0.55, zorder=1)
    axes[0].text(0.01, 0.04, "Symmetric noise", transform=axes[0].transAxes, color=INK2, fontsize=NOTE)
    legend_right(fig, axes, width=0.83)
    save(fig, "fig_noise")


def fig_bnlearn():
    """Methods by topology, raw and standardised: AncAcc centred on chance above, SHD per
    true edge below, so that topologies of different size share one colour scale."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("chance", ["#9a5328", "#f2f1ee", "#1f4a85"])
    # SHD: lower is better, so the good end keeps the blue of the AncAcc panels.
    cmap_shd = LinearSegmentedColormap.from_list("shd", ["#1f4a85", "#f2f1ee", "#9a5328"])
    d = load("P_bnlearn")
    d = d[d.K == 10]
    graphs = ["asia", "sachs", "child", "insurance", "water", "mildew", "alarm", "barley"]
    rows = [("anc_acc", TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0), cmap, r"AncAcc $\uparrow$"),
            ("lfix_shd_ratio", Normalize(vmin=0.0, vmax=2.0), cmap_shd, r"SHD / $|E|$ $\downarrow$")]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 8.6), sharey=True)
    for (metric, norm, cm, label), row in zip(rows, axes):
        for ax, s in zip(row, ["raw", "std"]):
            t = d[d.setting == s].pivot_table(index="method", columns="graph", values=metric)
            t = t.reindex(index=ROSTER, columns=graphs)
            if s == "std":
                t.loc["VarSort"] = np.nan               # every variance is one
            t["Mean"] = t.mean(axis=1)
            ax.imshow(t.values, cmap=cm, norm=norm, aspect="auto")
            for (r, c), v in np.ndenumerate(t.values):
                if np.isnan(v):
                    continue
                dark = abs(norm(v) - 0.5) > 0.27
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                        color="white" if dark else INK)
            ax.axvline(len(graphs) - 0.5, color="white", lw=3)     # the mean apart from the graphs
            ax.set_xticks(range(len(graphs) + 1))
            ax.set_xticklabels([g.capitalize() for g in graphs] + ["Mean"], rotation=35, ha="right",
                               rotation_mode="anchor")
            ax.set_yticks(range(len(ROSTER)))
            ax.set_yticklabels(ROSTER)
            ax.set_title(f"{label}, {SETTING_LABEL[s].lower()} data")
            ax.grid(False)
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)               # the same topologies as the row below
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    save(fig, "fig_bnlearn")


def fig_sample():
    """Standardised data against N: one noise family and mixed families side by side."""
    d = load("E_sample", "E_sample2")
    d = d[d.setting == "std"].copy()
    d["group"] = np.where(d.dist == "mixed", "mixed", "single")
    methods = _no_random(ROSTER)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.9), sharey=True)
    for ax, (g, title) in zip(axes, [("single", "Single noise families"), ("mixed", "Mixed families")]):
        lines(ax, d[d.group == g], "n", methods, logx=True)
        chance(ax)
        ax.set_title(title)
        ax.set_xlabel("Total sample size $N$")
    axes[0].set_ylabel(r"AncAcc $\uparrow$")
    legend_right(fig, axes, width=0.83)
    save(fig, "fig_sample")


def fig_cost():
    """Floats sent and running time against p, the two halves of the old cost table."""
    d = load("E_scale")
    d = d[d.K == 10]
    methods = FEDERATED
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.9), sharex=True)
    for ax, (val, label) in zip(axes, [("floats", "Floats sent by all clients"),
                                       ("wall", "Running time (s)")]):
        lines(ax, d, "p", methods, val=val, band=False)
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.set_xlabel("Variables $p$ ($K = 10$)")
        ax.set_xticks([10, 20, 30, 50, 80])
    legend_right(fig, axes, width=0.83)
    save(fig, "fig_cost")


def fig_privacy():
    """Gaussian mechanism against the budget, and the reconstruction error against cohort size."""
    from matplotlib.patches import Patch
    j = DEPLOYMENT
    dp = pd.concat([pd.read_csv(os.path.join(j, f)) for f in ("dp_one_round.csv", "dp_fedrcdx.csv")],
                   ignore_index=True)
    eps = [0.5, 1, 2, 5, 10, 20, np.inf]
    labels = ["0.5", "1", "2", "5", "10", "20", "None"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.9))
    ax = axes[0]
    for m in ["FedRCD-NI", "FedRCD-Def", "FedRCD-X", "FedHC"]:
        g = dp[dp.method == m].groupby("eps").AncAcc
        mu, se = g.mean().reindex(eps), g.sem().reindex(eps)
        st = STYLE[m]
        xs = np.arange(len(eps))
        ax.plot(xs, mu.values, color=st["color"], ls=st["ls"], marker=st["marker"], label=m,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.fill_between(xs, mu - se, mu + se, color=st["color"], alpha=0.13, lw=0)
    chance(ax)
    ax.axvline(len(eps) - 1.5, color=INK2, lw=0.8, alpha=0.55)
    ax.set_xticks(np.arange(len(eps)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Privacy budget $\varepsilon$ (None: no noise)")
    ax.set_ylabel(r"AncAcc $\uparrow$")
    ax.set_title("Gaussian mechanism")
    ax.grid(axis="x", visible=False)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h + [chance_handle()], lab + ["Chance"], loc="upper left", borderaxespad=0.3)

    ax = axes[1]
    co = pd.read_csv(os.path.join(j, "cohorts.csv"))
    g = co.groupby("cohort_size")
    moments = [("mean_relerr", "Mean", "#27518a", "o"), ("var_relerr", "Variance", "#a65f2e", "s"),
               ("skew_relerr", "Skewness", "#188768", "^"), ("kurt_relerr", "Kurtosis", "#922856", "v")]
    for col, lab, c, mk in moments:
        ax.plot(g[col].mean().index, g[col].mean().values, color=c, marker=mk, label=lab,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xticks(sorted(co.cohort_size.unique()))
    ax.set_xlabel("Cohort size $m$")
    ax.set_ylabel("Relative reconstruction error")
    ax.set_title("Cohort aggregation")
    ax.legend(loc="upper left", borderaxespad=0.3)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_privacy")



FIGURES = {
    "noise": fig_noise, "bnlearn": fig_bnlearn, "instrument": fig_instrument,
    "privacy": fig_privacy, "sample": fig_sample, "violations": fig_violations,
    "cost": fig_cost,
}

if __name__ == "__main__":
    setup()
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()

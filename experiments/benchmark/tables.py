"""LaTeX tables of the extended article, from the collected CSVs in results/.

    python -m experiments.benchmark.tables

Writes one .tex fragment per table to experiments/benchmark/tables/, plus a .csv with the
same numbers. The tables use booktabs and the method macros of the article.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "tables")

MACRO = {
    "FedRCD-NI": r"\fedrcdni", "FedRCD-Def": r"\fedrcddef", "FedRCD-X": r"\fedrcdx",
    "FedHC": r"\fedhc", "FedISHC": r"\fedishc", "FedISHC-4": r"\fedishcfour",
    "HC-LiNGAM": r"\hclingam", "HC": r"\hc", "DirectLiNGAM": r"\directlingam",
    "ICA-LiNGAM": r"\icalingam",
}


def load(exp):
    d = pd.read_csv(os.path.join(RESULTS, f"{exp}.csv"), low_memory=False)
    return d[d["error"].fillna("").astype(str).str.len() == 0]


def cell(x, digits):
    x = x.dropna()
    if x.empty:
        return "{--}"
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f"{x.mean():.{digits}f} \\pm {se:.{digits}f}"


def name(m):
    return MACRO.get(m, r"\textsc{" + m + "}")


def write(fname, tex, frame):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{fname}.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    frame.to_csv(os.path.join(OUT, f"{fname}.csv"))
    print("wrote", fname)

# Row groups for the extended article, in the order every table uses.
# The same methods as every figure. HC and HC-LiNGAM return the orders of FedRCD-X
# and FedRCD-NI on every draw, and FedISHC-4 is an ablation shown only where it is used.
ROLES = [["FedRCD-NI", "FedRCD-Def", "FedRCD-X"],
              ["FedHC", "FedISHC", "NOTEARS-ADMM"],
              ["DirectLiNGAM", "ICA-LiNGAM"],
              ["VarSort", "R2Sort", "Random"]]


def t_main():
    """Main grid, grouped rows, means only; the largest standard error goes in the caption."""
    d = load("P_main")
    d = d[(d.p == 20) & (d.K == 10)]
    metrics = [("anc_acc", "AncAcc", "{:.2f}", False), ("lfix_shd", "SHD", "{:.0f}", True),
               ("lfix_sid", "SID", "{:.0f}", True), ("lfix_edge_f1", "F1", "{:.2f}", False)]
    keys = [(s, m) for s in ("raw", "std") for m, *_ in metrics]
    g = d.groupby(["method", "setting"])
    mean = pd.DataFrame({k: g[k[1]].mean().xs(k[0], level="setting") for k in keys})
    se = pd.DataFrame({k: g[k[1]].sem().xs(k[0], level="setting") for k in keys})
    # Every variance is one after standardisation, so VarSort has nothing to sort there.
    for m, *_ in metrics:
        mean.loc["VarSort", ("std", m)] = np.nan
    fmts = [f for _ in ("raw", "std") for _, _, f, _ in metrics]
    lower = {k for k in keys if dict((m, lw) for m, _, _, lw in metrics)[k[1]]}
    # The text quotes the largest standard error; print it so it can be checked.
    print("tab_main: largest SE, AncAcc %.3f, F1 %.3f" % (
        se[[k for k in keys if k[1] == "anc_acc"]].max().max(),
        se[[k for k in keys if k[1] == "lfix_edge_f1"]].max().max()))
    lines = [r"\begin{table*}[tbp]",
             r"    \caption{Structure recovery on raw and standardised data ($p = 20$, $K = 10$, $N = 2000$, "
             r"eleven noise settings, ten seeds). $^\dagger$: centralised. Bold: best; underlined: best federated.} \label{tab:main}",
             r"    \centering", r"    \footnotesize", r"    \setlength{\tabcolsep}{3.6pt}",
             r"    \begin{tabular}{@{} l cccc cccc @{}}",
             r"        \toprule",
             r"        & \multicolumn{4}{c}{\textsc{\bfseries Raw data}} & \multicolumn{4}{c}{\textsc{\bfseries Standardised data}} \\",
             r"        \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"        \textsc{\bfseries Method} & " + " & ".join(f"{{\\textsc{{\\bfseries {h}}}}}" for _ in range(2) for _, h, _, _ in metrics) + r" \\",
             r"        \midrule"]
    lines += grouped_rows(mean, keys, fmts, lower, ROLES)
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}", ""]
    write("tab_main", "\n".join(lines), mean)


def grouped_rows(mean, keys, fmts, lower, roles, ablation=(), plain=()):
    """Grouped rows with split rules; bold best, underline best federated; nulls never marked.
    Columns in `plain` are descriptive (no best), so they are never marked either."""
    nulls, central = roles[-1], roles[-2]
    fed = [m for grp in roles[:-2] for m in grp if m not in ablation]
    best, bestfed = {}, {}
    for k, f in zip(keys, fmts):
        for store, pool in ((best, [m for m in mean.index if m not in nulls and m not in ablation]),
                            (bestfed, [m for m in mean.index if m in fed])):
            v = mean.loc[pool, k].dropna()
            if len(v):
                store[k] = f.format(v.min() if k in lower else v.max())
    out = []
    ncols = len(keys) + 1
    groups = [[m for m in grp if m in mean.index] for grp in roles]
    for gi, ms in enumerate([grp for grp in groups if grp]):
        if gi:
            out.append(f"        \\cmidrule(r){{1-1}}\\cmidrule(l){{2-{ncols}}}")
        for m in ms:
            cells = []
            for k, f in zip(keys, fmts):
                v = mean.loc[m, k]
                if pd.isna(v):
                    cells.append("--")
                    continue
                c = f.format(v).replace("-0.00", "0.00")
                if k in plain:
                    pass
                elif m not in nulls and m not in ablation and c == best.get(k):
                    c = f"\\textbf{{{c}}}"
                elif m in fed and c == bestfed.get(k) and best.get(k) != bestfed.get(k):
                    c = f"\\underline{{{c}}}"
                cells.append(c)
            # \xspace would put a space before the dagger, so close the macro with {}.
            label = name(m) + (r"{}$^\dagger$" if m in central else "")
            out.append(f"        {label} & " + " & ".join(cells) + r" \\")
    return out


def t_p():
    """AncAcc against p on raw data, with the FedISHC-4 ablation as its own group."""
    d = load("P_main")
    d = d[(d.K == 10) & (d.setting == "raw")]
    ps = sorted(d.p.unique())
    mean = d.pivot_table(index="method", columns="p", values="anc_acc")
    keys, fmts = ps, ["{:.2f}"] * len(ps)
    roles = [ROLES[0], ROLES[1], ["FedISHC-4"], ROLES[2], ROLES[3]]
    lines = [r"\begin{table}[tbp]",
             r"    \caption{\AncAcc against the number of variables $p$ on raw data ($K = 10$, $N = 2000$, "
             r"eleven noise settings, ten seeds). $^\dagger$: centralised. Bold: best; underlined: best federated.} \label{tab:p}",
             r"    \centering", r"    \footnotesize", r"    \setlength{\tabcolsep}{8pt}",
             r"    \begin{tabular}{@{} l " + "c" * len(ps) + r" @{}}",
             r"        \toprule",
             r"        \textsc{\bfseries Method} & " + " & ".join(f"$p = {p}$" for p in ps) + r" \\",
             r"        \midrule"]
    lines += grouped_rows(mean, keys, fmts, set(), roles, ablation=("FedISHC-4",))
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    write("tab_p", "\n".join(lines), mean)


def t_regime():
    """AncAcc by noise regime, raw and standardised: the PGM's stratification table."""
    d = load("P_main")
    d = d[(d.p == 20) & (d.K == 10)]
    regimes = [("symmetric", "Symmetric"), ("skewed", "Skewed"), ("mixed", "Mixed")]
    keys = [(s, r) for s in ("raw", "std") for r, _ in regimes]
    mean = d.pivot_table(index="method", columns=["setting", "regime"], values="anc_acc")[keys]
    for r, _ in regimes:
        mean.loc["VarSort", ("std", r)] = np.nan          # every variance is one
    lines = [r"\begin{table}[tbp]",
             r"    \caption{\AncAcc by noise regime on raw and standardised data ($p = 20$, $K = 10$, "
             r"$N = 2000$, ten seeds). $^\dagger$: centralised. Bold: best; underlined: best federated.} \label{tab:regime}",
             r"    \centering", r"    \footnotesize", r"    \setlength{\tabcolsep}{6pt}",
             r"    \begin{tabular}{@{} l ccc ccc @{}}",
             r"        \toprule",
             r"        & \multicolumn{3}{c}{\textsc{\bfseries Raw data}} & \multicolumn{3}{c}{\textsc{\bfseries Standardised data}} \\",
             r"        \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"        \textsc{\bfseries Method} & " + " & ".join(h for _ in range(2) for _, h in regimes) + r" \\",
             r"        \midrule"]
    lines += grouped_rows(mean, keys, ["{:.2f}"] * len(keys), set(), ROLES)
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    write("tab_regime", "\n".join(lines), mean)


def t_real():
    """Real data and the ladder diagnostic in one table: AncAcc raw and standardised, and
    Kendall's tau between the returned order and the variance sort, which needs no graph.
    tau is reported on raw data only: after standardisation every variance is one, and it
    would measure floating-point noise in the unit diagonal."""
    d = load("P_main")
    d = d[(d.p == 20) & (d.K == 10) & (d.setting == "raw")]
    r = load("E_real")
    sets = {"danube": r[(r.dataset == "danube_raw") & (r.K == 10)],
            "sachs": r[r.dataset == "sachs_conditions"]}
    cols = {("main", "tau"): d.groupby("method").ladder_agreement.mean()}
    for ds, x in sets.items():
        g = x.groupby(["method", "setting"])
        cols[(ds, "raw")] = g.anc_acc.mean().xs("raw", level="setting")
        cols[(ds, "std")] = g.anc_acc.mean().xs("std", level="setting")
        cols[(ds, "tau")] = x[x.setting == "raw"].groupby("method").ladder_agreement.mean()
    mean = pd.DataFrame(cols)
    for ds in sets:
        mean.loc["VarSort", (ds, "std")] = np.nan          # every variance is one
    keys = list(cols)
    taus = {k for k in keys if k[1] == "tau"}
    lines = [r"\begin{table}[tbp]",
             r"    \caption{\AncAcc on the real datasets, and Kendall's $\tau$ between the returned order and "
             r"the order by marginal variance on raw data (main grid: $p = 20$, $K = 10$, ten seeds; real data: "
             r"five seeds). $^\dagger$: centralised. Bold: best; underlined: best federated.} \label{tab:real}",
             r"    \centering", r"    \footnotesize", r"    \setlength{\tabcolsep}{5pt}",
             r"    \begin{tabular}{@{} l c ccc ccc @{}}",
             r"        \toprule",
             r"        & \textsc{\bfseries Main grid} & \multicolumn{3}{c}{\textsc{\bfseries Danube}} "
             r"& \multicolumn{3}{c}{\textsc{\bfseries Sachs}} \\",
             r"        \cmidrule(lr){2-2}\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"        \textsc{\bfseries Method} & $\tau$ & Raw & Std. & $\tau$ & Raw & Std. & $\tau$ \\",
             r"        \midrule"]
    lines += grouped_rows(mean, keys, ["{:.2f}"] * len(keys), set(), ROLES, plain=taus)
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    write("tab_real", "\n".join(lines), mean)


if __name__ == "__main__":
    t_main()
    t_p()
    t_regime()
    t_real()

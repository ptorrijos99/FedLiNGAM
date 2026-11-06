"""Run the experiments in parallel, one shard per task, resumable.

    python -m experiments.benchmark.run --exp P_main --jobs 20
    python -m experiments.benchmark.run --exp P_main --limit 4 --jobs 1     # smoke
    python -m experiments.benchmark.run --collect P_main
    python -m experiments.benchmark.run --exp P_main --chunk 3 --chunks 400 --jobs 1   # one SLURM array task

Each task writes `results/<exp>/<task>.csv` (one row per method x setting) and
`results/<exp>/<task>.npz` (every order and every estimated B), so no metric
added later ever needs a re-run. A task whose CSV exists is skipped.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# torch and MKL (numpy/scipy) each ship a libiomp5md.dll, and NOTEARS-ADMM calls
# scipy's expm inside torch autograd, which initialises both and aborts on Windows
# (OMP Error #15). The PGM benchmark ran with this flag set, because importing
# bnlearn sets it. With one thread per process the two runtimes never run a
# parallel region at once.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch                                           # noqa: E402,F401
    torch.set_num_threads(1)
except ImportError:
    pass

import argparse                                            # noqa: E402
import json                                                # noqa: E402
import sys                                                 # noqa: E402
import time                                                # noqa: E402
import traceback                                           # noqa: E402
import warnings                                            # noqa: E402
import zlib                                                # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Shards go outside the repository (thousands of small files); `--collect` writes
# the merged CSVs into experiments/benchmark/results/.
RESULTS = os.environ.get("FEDRCD_RESULTS", os.path.join(os.path.expanduser("~"), "fedrcd_results"))
COLLECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def task_id(task):
    cell = "_".join(f"{k}-{v}" for k, v in sorted(task["cell"].items()))
    return f"{cell}_seed-{task['seed']}".replace("/", "-")


def method_seed(task, method):
    """Stable across processes, unlike hash()."""
    return zlib.crc32(f"{task_id(task)}|{method}".encode()) % (2 ** 31)


def run_task(task):
    warnings.filterwarnings("ignore")
    import io
    from contextlib import redirect_stdout

    from experiments.benchmark import metrics as M
    from experiments.benchmark.data import standardise
    from experiments.benchmark.grids import make_data
    from experiments.benchmark.methods import REGISTRY, pooled, pooled_cov, reconstructions

    out_dir = os.path.join(RESULTS, task["exp"])
    os.makedirs(out_dir, exist_ok=True)
    tid = task_id(task)
    csv_path = os.path.join(out_dir, f"{tid}.csv")
    if os.path.exists(csv_path):
        return tid, "skipped", 0.0

    t_task = time.perf_counter()
    base = {"exp": task["exp"], "task": tid, "seed": task["seed"], **task["cell"]}
    rows, arrays = [], {}
    try:
        with redirect_stdout(io.StringIO()):
            clients, B_true, true_order, extra = make_data(task["cell"], task["seed"])
    except Exception as e:                                   # a failed draw is recorded, not fatal
        rows.append({**base, "method": "__data__", "error": f"{type(e).__name__}: {e}"})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return tid, "data-error", time.perf_counter() - t_task

    arrays["B_true"] = B_true
    base.update(extra)
    base["n_total"] = int(sum(len(c) for c in clients))
    base["n_clients_actual"] = len(clients)

    for setting in task["settings"]:
        data = {"raw": lambda c: c, "std": standardise}[setting](clients)
        X = pooled(data)
        cov = pooled_cov(data)
        n = len(X)
        diag = M.data_diagnostics(B_true, X, cov)
        for method in task["methods"]:
            fn, family = REGISTRY[method]
            row = {**base, **diag, "setting": setting, "method": method, "family": family}
            try:
                with redirect_stdout(io.StringIO()):
                    t0 = time.perf_counter()
                    res = fn(data, method_seed(task, method))
                    row["wall"] = time.perf_counter() - t0
                row.update(t_client=res.t_client, t_server=res.t_server,
                           floats=res.floats, rounds=res.rounds, **res.extra)
                row.update(M.order_metrics(B_true, res.order, true_order, cov))
                Bs = reconstructions(res.order, cov, n)
                if res.B_native is not None:
                    Bs["native"] = res.B_native
                for name, B in Bs.items():
                    row.update(M.graph_metrics(B_true, B, name))
                    arrays[f"{setting}|{method}|B_{name}"] = B
                arrays[f"{setting}|{method}|order"] = np.asarray(res.order)
                row["error"] = ""
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row["traceback"] = traceback.format_exc(limit=3)
            rows.append(row)

    np.savez_compressed(os.path.join(out_dir, f"{tid}.npz"), **arrays)
    tmp = csv_path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    return tid, "done", time.perf_counter() - t_task


def collect(exp):
    folder = os.path.join(RESULTS, exp)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    os.makedirs(COLLECTED, exist_ok=True)
    out = os.path.join(COLLECTED, f"{exp}.csv")
    df.to_csv(out, index=False)
    errors = df["error"].fillna("").astype(str).str.len().gt(0).sum() if "error" in df else 0
    print(f"{exp}: {len(files)} tasks, {len(df)} rows, {errors} errors -> {out}")
    return df


def main():
    from experiments.benchmark.grids import GRIDS

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=max(os.cpu_count() - 4, 1))
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke tests)")
    ap.add_argument("--chunk", type=int, default=None, help="SLURM array index: run tasks[chunk::chunks]")
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--count", action="store_true", help="print the number of tasks and exit")
    ap.add_argument("--collect", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, grid in GRIDS.items():
            print(f"{name:14s} {len(grid()):6d} tasks")
        return
    if args.collect is not None:
        for exp in args.collect or sorted(d for d in os.listdir(RESULTS) if os.path.isdir(os.path.join(RESULTS, d))):
            collect(exp)
        return

    tasks = [t for exp in args.exp for t in GRIDS[exp]()]
    if args.count:
        print(len(tasks))
        return
    if args.limit:
        tasks = tasks[: args.limit]
    if args.chunk is not None:
        # Round-robin, so the slow cells (NOTEARS-ADMM, large p) spread across chunks.
        tasks = tasks[args.chunk :: args.chunks]
    todo = [t for t in tasks if not os.path.exists(os.path.join(RESULTS, t["exp"], f"{task_id(t)}.csv"))]
    print(f"{len(tasks)} tasks, {len(todo)} to run, {args.jobs} workers", flush=True)

    t0 = time.time()
    if args.jobs == 1:
        for i, t in enumerate(todo, 1):
            tid, status, dt = run_task(t)
            print(f"[{i}/{len(todo)}] {status} {dt:.1f}s {tid}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_task, t) for t in todo]
            for i, f in enumerate(as_completed(futures), 1):
                try:
                    tid, status, dt = f.result()
                except Exception as e:
                    tid, status, dt = "?", f"crash {e}", 0.0
                if i % 25 == 0 or i == len(todo) or not status.startswith("done"):
                    el = time.time() - t0
                    print(f"[{i}/{len(todo)}] {el/60:.1f} min elapsed, ~{el/i*(len(todo)-i)/60:.0f} min left; "
                          f"last: {status} {dt:.1f}s {tid}", flush=True)
    print(json.dumps({"finished": len(todo), "minutes": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    main()

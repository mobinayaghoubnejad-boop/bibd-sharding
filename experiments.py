"""
experiments.py — اجرای همه آزمایش‌های فصل چهارم

نحوه اجرا (در ترمینال، داخل همین پوشه):
    python experiments.py main        # مقایسه اصلی با baselineها (۴ مقیاس × ۲ بار کاری × ۱۵ تکرار)
    python experiments.py ablation    # آزمایش حذف اجزای چهارنسخه‌ای با بودجه یکسان
    python experiments.py scale       # اجرای روی ۱۰۰٬۰۰۰ حساب
    python experiments.py ethereum --csv tx.csv --shards 13   # داده واقعی اتریوم
    python experiments.py quick       # اجرای سریع برای تست (چند ثانیه)

پروتکل هر اجرا (تقسیم زمانی، بخش ۳-۶):
    تراکنش‌ها به ترتیب زمان: ۷۰٪ اول = آموزش، ۳۰٪ آخر = آزمون
    epoch1 = نیمه اول آموزش ، epoch2 = کل آموزش
    sigma1 روی epoch1، sigma2 روی epoch2 (روش پیشنهادی از sigma1 ادامه می‌دهد)
    ρ_cross و CV_L روی پنجره آزمون؛ J، زمان، حافظه و مهاجرت برای epoch2
نتایج در پوشه results/ ذخیره می‌شوند.
"""
import argparse
import os
import time
import tracemalloc
import numpy as np
import pandas as pd

from designs import CyclicBIBD
from workload import random_workload, structured_workload, load_ethereum_csv, temporal_split, build_graph
from partition import bibd_method, hash_partition, ldg, louvain_partition, objective
from metrics import rho_cross, cv_load, align_labels, migration, compare

OUT = "results"
SCALES = [(28, 7), (78, 13), (186, 31), (310, 31)]   # (N, S) ؛ v = S
TX_PER_ACCOUNT = 20
SEEDS = 15
EPS = 0.05
BUDGET_PASSES = 10        # بودجه = BUDGET_PASSES × N × r  ارزیابی کاندید


def timed(fn):
    """اجرای fn و اندازه‌گیری زمان (ثانیه) و بیشینه حافظه (کیلوبایت)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, peak / 1024


def evaluate_methods(txs, N, S, methods, seed):
    """اجرای مجموعه‌ای از روش‌ها روی یک بار کاری با پروتکل دو-epoch و تقسیم زمانی."""
    train, test = temporal_split(txs, 0.7)
    e1 = train[: len(train) // 2]
    adj1, adj2 = build_graph(e1, N), build_graph(train, N)
    rows = []
    for name, run in methods.items():
        rng = np.random.default_rng(seed)
        sigma1, _ = run(adj1, None, rng)
        (sigma2, info), dt, mem = timed(lambda: run(adj2, sigma1, rng))
        if not (info or {}).get("warm", False):
            sigma2 = align_labels(sigma1, sigma2, S)
        rows.append({
            "method": name, "seed": seed, "N": N, "S": S,
            "rho_cross": rho_cross(test, sigma2),
            "CV_L": cv_load(test, sigma2, S),
            "rho_train": rho_cross(train, sigma2),
            "objective_J": objective(adj2, sigma2),
            "migration": migration(sigma1, sigma2),
            "time_ms": dt * 1000, "memory_KB": mem,
            "search_space": (info or {}).get("search_space", N * S),
            "evals": (info or {}).get("evals", 0),
        })
    return rows


def make_methods(N, S, design, ablation=False, with_louvain=True, seed=0):
    budget = BUDGET_PASSES * N * design.r

    def bibd_variant(mode, optimize=True):
        def run(adj, prev, rng):
            ws = None if prev is None else (prev, run.phi)
            sigma, info = bibd_method(adj, N, S, design, budget, EPS, mode=mode,
                                      optimize=optimize, rng=rng, warm_start=ws)
            run.phi = info["phi"]
            info["warm"] = prev is not None
            return sigma, info
        run.phi = None
        return run

    def wrap(f):
        return lambda adj, prev, rng: (f(adj), {"search_space": N * S})

    if ablation:
        return {
            "(1) BIBD no-opt": bibd_variant("bibd", optimize=False),
            "(2) Unconstrained opt": bibd_variant("none"),
            "(3) Random-constrained opt": bibd_variant("random"),
            "(4) BIBD-constrained opt": bibd_variant("bibd"),
        }
    m = {
        "Random/Hash": wrap(lambda adj: hash_partition(N, S)),
        "LDG": wrap(lambda adj: ldg(adj, N, S, EPS)),
    }
    if with_louvain:
        m["Community(Louvain)"] = wrap(lambda adj: louvain_partition(adj, N, S, EPS, seed))
    m["BIBD (proposed)"] = bibd_variant("bibd")
    return m


def gen(kind, N, rng):
    n_tx = TX_PER_ACCOUNT * N
    return structured_workload(N, n_tx, rng) if kind == "structured" else random_workload(N, n_tx, rng)


def summarize(df, name, ref="BIBD (proposed)"):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/{name}_raw.csv", index=False)
    keys = [k for k in ["workload", "N", "S"] if k in df.columns]
    agg = df.groupby(keys + ["method"]).agg(["mean", "std"]).drop(columns=["seed"], errors="ignore")
    agg.to_csv(f"{OUT}/{name}_summary.csv")
    print(f"\n==== {name}: mean (std) ====")
    cols = ["rho_cross", "CV_L", "migration", "time_ms", "memory_KB", "objective_J", "search_space"]
    mean = df.groupby(keys + ["method"])[cols].mean().round(4)
    print(mean.to_string())
    # مقایسه آماری روش پیشنهادی با هر روش دیگر
    comp = []
    for g, sub in df.groupby(keys):
        if ref not in set(sub.method):
            continue
        x = sub[sub.method == ref]
        for m in sorted(set(sub.method) - {ref}):
            y = sub[sub.method == m]
            for metric in ["rho_cross", "CV_L"]:
                if len(x) > 1 and len(y) > 1:
                    r = compare(x[metric], y[metric])
                    comp.append(dict(zip(keys, g if isinstance(g, tuple) else (g,)), vs=m, metric=metric, **r))
    if comp:
        cdf = pd.DataFrame(comp)
        cdf.to_csv(f"{OUT}/{name}_comparison.csv", index=False)
        print(f"\n---- {ref} در برابر سایر روش‌ها (بهبود مثبت = روش پیشنهادی بهتر) ----")
        print(cdf.round(4).to_string())


def run_main(scales=SCALES, seeds=SEEDS, workloads=("random", "structured")):
    rows = []
    for kind in workloads:
        for N, S in scales:
            design = CyclicBIBD(S)
            for seed in range(seeds):
                txs = gen(kind, N, np.random.default_rng(seed))
                for r in evaluate_methods(txs, N, S, make_methods(N, S, design, seed=seed), seed):
                    r["workload"] = kind
                    rows.append(r)
            print(f"done: {kind} N={N}")
    summarize(pd.DataFrame(rows), "main")
    plot_main(pd.DataFrame(rows))


def run_ablation(scales=SCALES, seeds=SEEDS):
    rows = []
    for N, S in scales:
        design = CyclicBIBD(S)
        for seed in range(seeds):
            txs = gen("structured", N, np.random.default_rng(seed))
            for r in evaluate_methods(txs, N, S, make_methods(N, S, design, ablation=True), seed):
                r["workload"] = "structured"
                rows.append(r)
        print(f"done ablation N={N}")
    summarize(pd.DataFrame(rows), "ablation", ref="(4) BIBD-constrained opt")


def run_scale(N=100_000, S=31, seeds=3, with_louvain=False):
    """اجرای روی ۱۰۰٬۰۰۰ حساب؛ فقط ρ_cross و CV_L (به‌علاوه زمان) گزارش می‌شوند."""
    design = CyclicBIBD(S)
    rows = []
    for seed in range(seeds):
        txs = structured_workload(N, 10 * N, np.random.default_rng(seed))
        for r in evaluate_methods(txs, N, S, make_methods(N, S, design, with_louvain=with_louvain), seed):
            r["workload"] = "structured"
            rows.append(r)
        print(f"done scale seed={seed}")
    summarize(pd.DataFrame(rows), "scale100k")


def run_ethereum(csv_path, S=13, max_accounts=None, **cols):
    txs, N, _ = load_ethereum_csv(csv_path, max_accounts, **cols)
    print(f"Ethereum: {len(txs)} تراکنش، {N} حساب")
    design = CyclicBIBD(S)
    rows = evaluate_methods(txs, N, S, make_methods(N, S, design), seed=0)
    for r in rows:
        r["workload"] = "ethereum"
    summarize(pd.DataFrame(rows), "ethereum")


def plot_main(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind, sub in df.groupby("workload"):
        for metric in ["rho_cross", "CV_L"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            g = sub.groupby(["method", "N"])[metric].agg(["mean", "std"]).reset_index()
            for m, gg in g.groupby("method"):
                ax.errorbar(gg["N"], gg["mean"], yerr=gg["std"], marker="o", capsize=3, label=m)
            ax.set_xscale("log")
            ax.set_xlabel("N (accounts)")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} - {kind} workload")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(f"{OUT}/fig_{metric}_{kind}.png", dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["main", "ablation", "scale", "ethereum", "quick"])
    p.add_argument("--csv")
    p.add_argument("--shards", type=int, default=13)
    p.add_argument("--max-accounts", type=int)
    p.add_argument("--from-col", default="from_address")
    p.add_argument("--to-col", default="to_address")
    p.add_argument("--time-col", default="block_number")
    a = p.parse_args()
    if a.mode == "main":
        run_main()
    elif a.mode == "ablation":
        run_ablation()
    elif a.mode == "scale":
        run_scale()
    elif a.mode == "ethereum":
        run_ethereum(a.csv, a.shards, a.max_accounts, from_col=a.from_col, to_col=a.to_col, time_col=a.time_col)
    elif a.mode == "quick":
        run_main(scales=[(28, 7), (78, 13)], seeds=3)
        run_ablation(scales=[(78, 13)], seeds=3)

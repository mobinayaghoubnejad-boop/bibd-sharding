"""
txallo_runs.py — TxAllo (ICDE'23) under the paper's two-epoch protocol, same workloads and seeds.

  python txallo_runs.py main                      # 4 scales x {structured, random} x 15 seeds
  python txallo_runs.py scale                     # N = 100,000, 3 seeds
  python txallo_runs.py ethereum --csv eth.csv    # S = 13 and 31, 15 seeds

G-TxAllo rebuilds at every epoch; A-TxAllo updates the epoch-1 allocation with the accounts of the
new transactions. "-C" variants respect the account capacity ceil((1+eps)N/S) of Eq. (2).
Extra columns: max_load_ratio = largest shard size / capacity, empty_shards.
"""
import argparse, os
import numpy as np, pandas as pd
import experiments as e
from workload import structured_workload, load_ethereum_csv
from partition import capacity
from txallo import make_txallo_methods

OUT = "results"


def methods(N, S, seed):
    m = make_txallo_methods(N, S, seed)
    m.update(make_txallo_methods(N, S, seed, constrained=True))
    return m


def run(txs, N, S, seed, tag):
    # evaluate_methods returns only metrics; recompute shard sizes from a second pass is costly,
    # so wrap each method to record the final allocation.
    ms, final = methods(N, S, seed), {}
    def keep(name, f):
        def g(adj, prev, rng):
            sig, info = f(adj, prev, rng)
            if prev is not None:
                final[name] = sig
            return sig, info
        return g
    ms = {k: keep(k, f) for k, f in ms.items()}
    rows = e.evaluate_methods(txs, N, S, ms, seed)
    cap = capacity(N, S, e.EPS)
    for r in rows:
        sz = np.bincount(final[r["method"]], minlength=S)
        r.update(workload=tag, max_load_ratio=sz.max() / cap, empty_shards=int((sz == 0).sum()))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["main", "scale", "ethereum"])
    p.add_argument("--csv")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    if a.mode == "main":
        for kind in ("structured", "random"):
            for N, S in e.SCALES:
                for seed in range(e.SEEDS):
                    rows += run(e.gen(kind, N, np.random.default_rng(seed)), N, S, seed, kind)
                print("done", kind, N, flush=True)
    elif a.mode == "scale":
        for seed in range(3):
            N = 100_000
            rows += run(structured_workload(N, 10 * N, np.random.default_rng(seed)), N, 31, seed, "structured")
            print("done scale", seed, flush=True)
    else:
        txs, N, _ = load_ethereum_csv(a.csv)
        for S in (13, 31):
            for seed in range(15):
                rows += run(txs, N, S, seed, "ethereum")
            print("done eth", S, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/txallo_{a.mode}_raw.csv", index=False)
    print("FINISHED")

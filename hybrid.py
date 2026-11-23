"""
hybrid.py — multi-epoch experiment: Louvain at epoch 0, BIBD-constrained warm-started updates at epochs 1..E-1.

The transaction stream is cut into E+1 consecutive, non-overlapping windows W_0..W_E.
The partition of epoch e is built on W_e and evaluated (rho_cross, CV_L) on W_{e+1}.
Migration of epoch e (e >= 1) = share of accounts whose shard differs from epoch e-1.

Strategies
  Louvain   : rebuild with Louvain at every epoch (labels aligned to previous epoch, Hungarian)
  LDG       : rebuild with LDG at every epoch (aligned)
  Hash      : static
  BIBD      : proposed method (LDG grouping + local search at epoch 0, warm start afterwards)
  Hybrid    : Louvain at epoch 0; groups phi(a) = (sigma_L(a) + d0) mod v, so that
              sigma_L(a) = pi(phi(a)) lies in Cand(a) (inverse of the matching in Prop. 3);
              BIBD-constrained warm-started local search at epochs 1..E-1.

Usage:
  python hybrid.py synthetic   [--seeds 15]
  python hybrid.py ethereum --csv eth.csv [--seeds 15]
  python hybrid.py scale       [--seeds 3]
"""
import argparse
import os
import time
import numpy as np
import pandas as pd

from designs import CyclicBIBD
from workload import structured_workload, load_ethereum_csv, build_graph
from partition import bibd_method, hash_partition, ldg, louvain_partition
from metrics import rho_cross, cv_load, align_labels, migration
from txallo import g_txallo, g_txallo_c, a_txallo
from partition import capacity

EPS = 0.05
BUDGET_PASSES = 10
EPOCHS = 6            # epochs 0..5
OUT = "results"


def windows(txs, n):
    txs = sorted(txs, key=lambda x: x[2])
    L = len(txs)
    return [txs[i * L // n:(i + 1) * L // n] for i in range(n)]


def run_stream(wins, N, S, seed, tag):
    design = CyclicBIBD(S)
    budget = BUDGET_PASSES * N * design.r
    d0 = design.D[0]
    adjs = [build_graph(w, N) for w in wins[:EPOCHS]]
    rows = []

    def record(method, e, sigma, prev, dt):
        rows.append({
            "set": tag, "method": method, "seed": seed, "N": N, "S": S, "epoch": e,
            "rho_cross": rho_cross(wins[e + 1], sigma),
            "CV_L": cv_load(wins[e + 1], sigma, S),
            "migration": migration(prev, sigma) if prev is not None else np.nan,
            "time_s": dt,
        })

    # --- rebuild-from-scratch baselines
    for name in ("Louvain", "LDG"):
        prev = None
        for e in range(EPOCHS):
            t0 = time.perf_counter()
            if name == "Louvain":
                sig = louvain_partition(adjs[e], N, S, EPS, seed=seed * 100 + e)
            else:
                sig = ldg(adjs[e], N, S, EPS)
            dt = time.perf_counter() - t0
            if name == "Louvain" and e == 0:
                lv0 = (sig.copy(), dt)
            if prev is not None:
                sig = align_labels(prev, sig, S)
            record(name, e, sig, prev, dt)
            prev = sig

    # --- hash (static)
    sig = hash_partition(N, S)
    for e in range(EPOCHS):
        record("Hash", e, sig, sig if e else None, 0.0)

    # --- proposed BIBD
    rng = np.random.default_rng(seed)
    prev, phi = None, None
    for e in range(EPOCHS):
        t0 = time.perf_counter()
        ws = None if prev is None else (prev, phi)
        sig, info = bibd_method(adjs[e], N, S, design, budget, EPS, rng=rng, warm_start=ws)
        dt = time.perf_counter() - t0
        phi = info["phi"]
        record("BIBD", e, sig, prev, dt)
        prev = sig

    # --- hybrid: Louvain at epoch 0, BIBD updates afterwards
    rng = np.random.default_rng(seed)
    sig, dt = lv0                     # same Louvain run as the Louvain baseline at epoch 0
    phi = (sig + d0) % design.v
    record("Hybrid", 0, sig, None, dt)
    prev = sig
    for e in range(1, EPOCHS):
        t0 = time.perf_counter()
        sig, info = bibd_method(adjs[e], N, S, design, budget, EPS, rng=rng, warm_start=(prev, phi))
        dt = time.perf_counter() - t0
        record("Hybrid", e, sig, prev, dt)
        prev = sig

    # --- TxAllo (ICDE'23): G-TxAllo at epoch 0, A-TxAllo at epochs 1..E-1 (as published,
    #     and adapted to the account-capacity constraint), plus a hybrid that starts from
    #     the constrained G-TxAllo and continues with BIBD-constrained updates.
    cap = capacity(N, S, EPS)
    for name, G, c in (("TxAllo", lambda a: g_txallo(a, N, S, seed=seed), None),
                       ("TxAllo-C", lambda a: g_txallo_c(a, N, S, seed=seed), cap)):
        seen = np.zeros(N, dtype=bool)
        t0 = time.perf_counter()
        sig, _ = G(adjs[0])
        dt = time.perf_counter() - t0
        if c is not None:
            txc0 = (sig.copy(), dt)
        record(name, 0, sig, None, dt)
        seen |= np.array([bool(nb) for nb in adjs[0]])
        prev = sig
        for e in range(1, EPOCHS):
            vhat = [v for v in range(N) if adjs[e][v]]
            t0 = time.perf_counter()
            sig, _ = a_txallo(adjs[e], N, S, prev, vhat, seen, cap=c)
            dt = time.perf_counter() - t0
            record(name, e, sig, prev, dt)
            seen |= np.array([bool(nb) for nb in adjs[e]])
            prev = sig

    rng = np.random.default_rng(seed)
    sig, dt = txc0
    phi = (sig + d0) % design.v
    record("Hybrid-TxAllo", 0, sig, None, dt)
    prev = sig
    for e in range(1, EPOCHS):
        t0 = time.perf_counter()
        sig, info = bibd_method(adjs[e], N, S, design, budget, EPS, rng=rng, warm_start=(prev, phi))
        dt = time.perf_counter() - t0
        record("Hybrid-TxAllo", e, sig, prev, dt)
        prev = sig
    return rows


def synthetic(seeds, scales, tx_per_window):
    rows = []
    for N, S in scales:
        for seed in range(seeds):
            txs = structured_workload(N, (EPOCHS + 1) * tx_per_window * N, np.random.default_rng(seed))
            rows += run_stream(windows(txs, EPOCHS + 1), N, S, seed, "structured")
            print("done", N, seed, flush=True)
    return rows


def ethereum(csv, seeds):
    txs, N, _ = load_ethereum_csv(csv)
    wins = windows(txs, EPOCHS + 1)
    rows = []
    for S in (13, 31):
        for seed in range(seeds):
            rows += run_stream(wins, N, S, seed, "ethereum")
            print("done eth", S, seed, flush=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["synthetic", "ethereum", "scale"])
    p.add_argument("--csv")
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--prefix", default="hybrid")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == "synthetic":
        rows = synthetic(a.seeds, [(186, 31), (310, 31)], 10)
    elif a.mode == "scale":
        rows = synthetic(a.seeds, [(100_000, 31)], 5)
    else:
        rows = ethereum(a.csv, a.seeds)
    pd.DataFrame(rows).to_csv(f"{OUT}/{a.prefix}_{a.mode}_raw.csv", index=False)
    print("FINISHED")

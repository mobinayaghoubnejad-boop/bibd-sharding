# BIBD-Constrained Account Partitioning for State Sharding

Simulator and experiment scripts for the manuscript *"BIBD-Constrained Account Partitioning for State Sharding"*. The code reproduces every table and figure in the paper from synthetic workloads and a public Ethereum mainnet trace.

## Files

| File | Purpose | Paper section |
|---|---|---|
| `designs.py` | Cyclic symmetric BIBDs (Fano plane (7,7,3,3,1), (13,13,4,4,1), Paley (31,31,15,15,7)) and checks of their properties | 3, 4.2–4.3 |
| `workload.py` | Random and structured workloads, Ethereum CSV loader, temporal split, transaction graph G = (A, E, w) | 5.1–5.2 |
| `partition.py` | Proposed method (Algorithm 1) and baselines: Hash, LDG, Louvain | 4.4, 5.3 |
| `metrics.py` | Cross-shard ratio ρ_cross, load imbalance CV_L, migration, statistics (Welch t-test, Cohen's d, 95% CI, Holm correction) | 5.4, 5.6 |
| `experiments.py` | Main comparison, ablation, 100,000-account scale run, figures | 6.1–6.5 |
| `ethereum_full.py` | Full Ethereum protocol (S = 13 and 31, 15 runs, including ablation) | 6.6 |
| `hybrid.py` | Multi-epoch experiment: Louvain or TxAllo at epoch 0, BIBD-constrained warm-started updates at epochs 1–5 | 5.2, 6.9, Table 11 |
| `txallo.py` | Re-implementation of TxAllo (Zhang et al., ICDE 2023): G-TxAllo, A-TxAllo and the capacity-constrained variants | 5.3 |
| `txallo_runs.py` | TxAllo under the two-epoch protocol (synthetic, 100,000 accounts, Ethereum) | 6.7, Table 9 |
| `fig7.py` | Plots Fig. 7 from the output of `hybrid.py` | 6.9, Fig. 7 |
| `results/hybrid_*_raw.csv` | Raw results of the multi-epoch experiment (every method, run and epoch) | 6.9, Table 11 |
| `results/txallo_*_raw.csv` | Raw results of TxAllo under the two-epoch protocol | 6.7, Table 9 |
| `job_h40qzHAg_I74wwDbxiDffPbBuROA (1).csv` | Ethereum mainnet trace used in the paper (blocks 18,000,000–18,000,250) | 5.1, 6.6 |

## Requirements

Python 3.9 or later.

```
pip install numpy scipy pandas networkx matplotlib
```

## Usage

Rename the Ethereum trace to `eth.csv` (or pass its path to `--csv`).

```
python experiments.py quick       # smoke test (about one minute)
python experiments.py main        # main comparison: 4 scales x 2 workloads x 15 runs (Sections 6.1-6.2)
python experiments.py ablation    # four-variant ablation with equal budget (Section 6.3)
python experiments.py scale       # 100,000 accounts (Section 6.5)
python experiments.py ethereum --csv eth.csv --shards 13      # single Ethereum run
python ethereum_full.py . eth.csv eth_raw.csv                 # full Ethereum protocol (Section 6.6)
```

Outputs are written to `results/`: `*_raw.csv` (all runs), `*_summary.csv` (mean and standard deviation), `*_comparison.csv` (statistical tests) and `fig_*.png`.

## Proposed method

- **Objective:** minimize J(σ), the total weight of transactions whose two accounts lie in different shards.
- **Constraint 1 (BIBD):** σ(a) ∈ Cand(a); each account has exactly r admissible shards.
- **Constraint 2 (capacity):** every shard holds at most ⌈(1 + ε)N/S⌉ accounts, with ε = 0.05.
- **Procedure:**
  1. Group accounts into v groups with LDG (transaction-aware grouping), giving φ(a).
  2. Cand(a) = {(φ(a) − d) mod v : d ∈ D}, where D is the difference set of the design.
  3. Initial partition σ0(a) = π(φ(a)) with π(i) = (i − d0) mod v, a perfect matching (Proposition 3).
  4. Local search: move each account to the candidate shard with the largest reduction of J, if capacity allows.
  5. Stop when the budget B is exhausted or a full pass makes no move.
- **Budget:** B = 10 · N · r candidate evaluations. All ablation variants use the same budget.
- **Later epochs** start from the previous partition, which keeps migration low.

## Ablation (Section 6.3)

| Variant | Description |
|---|---|
| (1) BIBD, no optimization | initial partition σ0 only |
| (2) Unconstrained | same local search, every account may move to any of the S shards |
| (3) Random candidates | r random candidate shards per group, no BIBD structure |
| (4) BIBD-constrained | proposed method |

Reported metrics: `rho_cross`, `CV_L`, `migration`, `time_ms`, `memory_KB`, `objective_J` and `search_space`. Comparing (2) and (4) measures the quality cost of the BIBD constraint; comparing (3) and (4) separates the effect of the design structure from that of a smaller search space.

## Evaluation protocol (Section 5.2)

Transactions are sorted by time. The first 70% form the training window and the last 30% the test window, on which ρ_cross and CV_L are measured. The partition is built in two epochs (epoch 1 on the first half of the training window, epoch 2 on the whole training window), so that migration can be measured.

## TxAllo baseline (Sections 5.3 and 6.7)

TxAllo has no public code; `txallo.py` follows Section V of Zhang, Pan and Yu, "TxAllo: Dynamic Transaction Allocation in Sharded Blockchain Systems", ICDE 2023 (arXiv:2212.11584), with the parameters of its case study (η = 2, λ = |T|/k, ε = 1e-5·|T|).

| Method | Description |
|---|---|
| `G-TxAllo` | global algorithm: Louvain communities, keep the k heaviest, place the rest by throughput gain, then move accounts while the gain is positive |
| `A-TxAllo` | adaptive algorithm: start from the previous allocation and move only the accounts of the new transactions |
| `G-TxAllo-C`, `A-TxAllo-C` | same, with the account capacity ceil((1+ε)N/S) of Eq. (2): Louvain communities packed under capacity, no moves into full shards |

```
python txallo_runs.py main                     # 4 scales x {structured, random} x 15 seeds
python txallo_runs.py scale                    # 100,000 accounts, 3 seeds
python txallo_runs.py ethereum --csv eth.csv   # S = 13 and 31, 15 seeds
```

Outputs `results/txallo_{main,scale,ethereum}_raw.csv`, with the extra columns `max_load_ratio` (largest shard size / capacity) and `empty_shards`.

## Multi-epoch experiment with hybrids (Section 6.9)

The transaction stream is cut into seven consecutive, non-overlapping windows W0, …, W6. The partition of epoch e is built on We and evaluated on W(e+1); migration at epoch e is measured against epoch e − 1.

| Method | Description |
|---|---|
| `Louvain` | rebuilt at every epoch (labels aligned to the previous epoch with the Hungarian algorithm) |
| `LDG` | rebuilt at every epoch (aligned) |
| `Hash` | static |
| `BIBD` | proposed method: LDG grouping at epoch 0, warm start afterwards |
| `Hybrid` | Louvain at epoch 0; groups set to φ(a) = (σ_L(a) + d0) mod v, so the Louvain partition lies in the BIBD candidate space without moving any account; BIBD-constrained warm-started local search at epochs 1–5 |
| `TxAllo`, `TxAllo-C` | G-TxAllo at epoch 0, A-TxAllo at epochs 1–5 (as published, and with the capacity constraint) |
| `Hybrid-TxAllo` | G-TxAllo-C at epoch 0, mapped into the candidate space as above; BIBD-constrained updates at epochs 1–5 |

```
python hybrid.py synthetic --seeds 15                 # N = 186 and 310, S = 31, 10N transactions per window
python hybrid.py scale --seeds 3                      # N = 100,000, 5N transactions per window (about 25 minutes)
python hybrid.py ethereum --csv eth.csv --seeds 15    # Ethereum trace, S = 13 and 31
python fig7.py                                        # writes results/fig7_hybrid.png
```

Outputs: `results/hybrid_synthetic_raw.csv`, `results/hybrid_scale_raw.csv` and `results/hybrid_ethereum_raw.csv`. Each row is one method, run and epoch, with columns `rho_cross`, `CV_L`, `migration` and `time_s`. Table 11 reports the mean over epochs 1–5.

## Ethereum data

The trace was exported from Google BigQuery with:

```sql
SELECT from_address, to_address, block_number
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE block_number BETWEEN 18000000 AND 18000250
  AND to_address IS NOT NULL
```

After removing contract creations and self-transfers, these 251 blocks contain 35,531 transactions among 23,676 accounts. For other datasets (for example XBlock-ETH), set the column names with `--from-col`, `--to-col` and `--time-col`.

## Reproducibility notes

- Every run records its seed (0–14); results are fully reproducible.
- Time and memory are measured with `tracemalloc`, which adds a small overhead that is the same for all methods.
- Shard load is the number of transactions in which the shard participates; a cross-shard transaction counts for both shards.
- In the structured workload, hot accounts follow a power law, so CV_L is inherently high.

## License and citation

If you use this code, please cite the paper (citation will be added after publication).

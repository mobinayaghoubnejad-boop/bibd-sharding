# Reproduces Tables 8 and the Ethereum ablation: python ethereum_full.py <code_dir> <eth.csv> <out.csv>
import sys, pandas as pd
sys.path.insert(0, sys.argv[1])
import experiments as e
from designs import CyclicBIBD
from workload import load_ethereum_csv
txs, N, _ = load_ethereum_csv(sys.argv[2])
rows=[]
for S in (13, 31):
    d = CyclicBIBD(S)
    for seed in range(15):
        for abl in (False, True):
            for r in e.evaluate_methods(txs, N, S, e.make_methods(N, S, d, ablation=abl, seed=seed), seed):
                r["set"] = "ablation" if abl else "main"; rows.append(r)
        print("done", S, seed, flush=True)
pd.DataFrame(rows).to_csv(sys.argv[3], index=False)
print("FINISHED")

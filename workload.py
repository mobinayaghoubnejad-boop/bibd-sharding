"""
workload.py — تولید بار کاری مصنوعی و خواندن داده واقعی اتریوم

خروجی همه توابع: لیستی از تراکنش‌ها به شکل (a, b, t)
  a, b : شماره حساب‌ها (0..N-1)   t : زمان یا شماره بلوک (برای تقسیم زمانی)
"""
import csv
import numpy as np


def random_workload(N, n_tx, rng):
    """بار کاری تصادفی (شاهد): هر تراکنش دو حساب را یکنواخت و مستقل انتخاب می‌کند."""
    a = rng.integers(0, N, n_tx)
    b = rng.integers(0, N - 1, n_tx)
    b = np.where(b >= a, b + 1, b)          # a != b
    return [(int(x), int(y), i) for i, (x, y) in enumerate(zip(a, b))]


def structured_workload(N, n_tx, rng, n_comm=None, p_in=0.8, alpha=1.2, p_repeat=0.3):
    """
    بار کاری ساختاریافته با سه ویژگی:
      - ساختار جامعه‌ای: با احتمال p_in طرف دوم تراکنش از همان جامعه انتخاب می‌شود
      - حساب‌های داغ (توزیع توانی): احتمال انتخاب هر حساب ∝ rank^-alpha
      - محلی‌بودن زمانی: با احتمال p_repeat یکی از ۵۰ تعامل اخیر تکرار می‌شود
    """
    if n_comm is None:
        n_comm = max(2, N // 8)
    comm = rng.integers(0, n_comm, N)                      # جامعه هر حساب
    pop = 1.0 / (rng.permutation(N) + 1.0) ** alpha         # محبوبیت توانی
    all_p = pop / pop.sum()
    order = np.argsort(comm, kind="stable")                  # حساب‌ها مرتب بر اساس جامعه
    starts = np.searchsorted(comm[order], np.arange(n_comm + 1))
    cum = []                                                 # توزیع تجمعی درون هر جامعه
    for c in range(n_comm):
        m = order[starts[c]:starts[c + 1]]
        cum.append((m, np.cumsum(pop[m]) / pop[m].sum() if len(m) else None))
    A = rng.choice(N, size=n_tx, p=all_p)
    Bglob = rng.choice(N, size=n_tx, p=all_p)
    U = rng.random((n_tx, 3))
    recent, txs = [], []
    for t in range(n_tx):
        if recent and U[t, 0] < p_repeat:
            a, b = recent[int(U[t, 1] * len(recent))]
        else:
            a = int(A[t])
            m, cp = cum[comm[a]]
            if U[t, 1] < p_in and len(m) > 1:
                b = int(m[min(np.searchsorted(cp, U[t, 2]), len(m) - 1)])
            else:
                b = int(Bglob[t])
            if a == b:
                b = (a + 1) % N
        txs.append((a, b, t))
        recent.append((a, b))
        if len(recent) > 50:
            recent.pop(0)
    return txs


def load_ethereum_csv(path, max_accounts=None, from_col="from_address",
                      to_col="to_address", time_col="block_number"):
    """
    خواندن تراکنش‌های واقعی اتریوم از CSV (مثلاً خروجی Google BigQuery یا XBlock-ETH).
    نام ستون‌ها قابل تنظیم است. تراکنش‌های ایجاد قرارداد (to خالی) و خودارجاع حذف می‌شوند.
    اگر max_accounts داده شود، فقط فعال‌ترین حساب‌ها نگه داشته می‌شوند.
    خروجی: (txs, N, addresses)
    """
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            a, b = (r.get(from_col) or "").lower(), (r.get(to_col) or "").lower()
            if not a or not b or a == b:
                continue
            rows.append((a, b, int(float(r[time_col]))))
    rows.sort(key=lambda x: x[2])
    if max_accounts:
        from collections import Counter
        cnt = Counter()
        for a, b, _ in rows:
            cnt[a] += 1
            cnt[b] += 1
        keep = {x for x, _ in cnt.most_common(max_accounts)}
        rows = [r for r in rows if r[0] in keep and r[1] in keep]
    idx = {}
    for a, b, _ in rows:
        idx.setdefault(a, len(idx))
        idx.setdefault(b, len(idx))
    txs = [(idx[a], idx[b], t) for a, b, t in rows]
    return txs, len(idx), list(idx)


def temporal_split(txs, train_frac=0.7):
    """تقسیم زمانی: تراکنش‌های قدیمی‌تر برای ساخت افراز، جدیدتر برای سنجش."""
    txs = sorted(txs, key=lambda x: x[2])
    cut = int(len(txs) * train_frac)
    return txs[:cut], txs[cut:]


def build_graph(txs, N):
    """گراف وزن‌دار G=(A,E,w) به‌صورت لیست مجاورت: adj[a] = {b: w(a,b)}"""
    adj = [dict() for _ in range(N)]
    for a, b, _ in txs:
        adj[a][b] = adj[a].get(b, 0) + 1
        adj[b][a] = adj[b].get(a, 0) + 1
    return adj

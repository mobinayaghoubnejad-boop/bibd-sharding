"""
partition.py — روش پیشنهادی و baselineها

صورت‌بندی (مطابق بخش ۳-۲-۴ پایان‌نامه):
    min_sigma  J(sigma) = sum_{(a,b) in E} w(a,b) * 1[sigma(a) != sigma(b)]
    s.t.  sigma(a) in Cand(a)                  (قید ساختاری BIBD)
          |sigma^-1(s)| <= ceil((1+eps) N/S)  (قید ظرفیت)

الگوریتم پیشنهادی (BIBD-constrained local search):
    ۱) گروه‌بندی phi: هر حساب به یکی از v نقطه طرح (گروه‌های هم‌اندازه)
    ۲) Cand(a) = Cand(phi(a)) از روی مجموعه تفاضلی (r شارد)
    ۳) افراز اولیه شدنی sigma0(a) = pi(phi(a))  (تطابق کامل نقطه->بلوک)
    ۴) بهینه‌سازی محلی: جابه‌جایی تک‌حساب به بهترین شارد کاندید با بهره مثبت،
       با رعایت ظرفیت، تا پایان بودجه B یا نبود بهبود.
    بودجه B = تعداد «ارزیابی کاندید» (هر بار که بهره انتقال یک حساب به یک شارد حساب می‌شود).
    چون همه نسخه‌های ablation از یک sigma0 و یک بودجه شروع می‌کنند، مقایسه منصفانه است.
"""
import math
import numpy as np
import networkx as nx


# ---------------------------------------------------------------- کمکی‌ها
def capacity(N, S, eps):
    return math.ceil((1 + eps) * N / S)


def objective(adj, sigma):
    """J(sigma): وزن کل یال‌های بریده‌شده (هر یال یک بار شمرده می‌شود)."""
    cut = 0
    for a, nb in enumerate(adj):
        sa = sigma[a]
        for b, w in nb.items():
            if b > a and sigma[b] != sa:
                cut += w
    return cut


# ---------------------------------------------------------------- گروه‌بندی phi
def grouping_random(N, v, rng):
    """گروه‌بندی تصادفی متوازن (اندازه گروه‌ها حداکثر یک واحد اختلاف)."""
    perm = rng.permutation(N)
    phi = np.empty(N, dtype=int)
    phi[perm] = np.arange(N) % v
    return phi


def grouping_ldg(adj, N, v, rng):
    """گروه‌بندی آگاه از تراکنش با LDG به v گروه (نگاشت تراکنش‌محور)."""
    return ldg(adj, N, v, eps=0.0, rng=rng)


# ---------------------------------------------------------------- روش پیشنهادی و نسخه‌های ablation
def build_candidates(design, phi, mode, S, rng):
    """
    mode = 'bibd'   : Cand از طرح (r شارد)
    mode = 'random' : برای هر گروه r شارد تصادفی (شامل شارد اولیه) — کنترل ablation
    mode = 'none'   : همه S شارد (بهینه‌سازی بدون قید)
    خروجی: لیست کاندیدهای هر گروه
    """
    pi = design.initial_matching()
    v, r = design.v, design.r
    if mode == "bibd":
        return [design.candidates(i) for i in range(v)]
    if mode == "random":
        out = []
        for i in range(v):
            others = [s for s in range(S) if s != pi[i]]
            out.append(sorted([pi[i]] + list(rng.choice(others, r - 1, replace=False))))
        return out
    if mode == "none":
        return [list(range(S)) for _ in range(v)]
    raise ValueError(mode)


def initial_partition(design, phi):
    pi = design.initial_matching()
    return np.array([pi[g] for g in phi], dtype=int)


def local_search(adj, sigma, group_cands, phi, S, cap, budget, rng):
    """
    بهینه‌سازی محلی تک‌حساب با بودجه مشخص.
    خروجی: sigma بهینه‌شده، تعداد ارزیابی مصرف‌شده، تعداد جابه‌جایی
    """
    sigma = sigma.copy()
    N = len(sigma)
    sizes = np.bincount(sigma, minlength=S)
    evals = moves = 0
    while evals < budget:
        moved_in_pass = 0
        for a in rng.permutation(N):
            if evals >= budget:
                break
            nb = adj[a]
            if not nb:
                continue
            s = sigma[a]
            w_to = {}
            for b, w in nb.items():
                t = sigma[b]
                w_to[t] = w_to.get(t, 0) + w
            stay = w_to.get(s, 0)
            best_t, best_gain = s, 0
            for t in group_cands[phi[a]]:
                if t == s:
                    continue
                evals += 1
                gain = w_to.get(t, 0) - stay
                if gain > best_gain and sizes[t] < cap:
                    best_t, best_gain = t, gain
            if best_t != s:
                sigma[a] = best_t
                sizes[s] -= 1
                sizes[best_t] += 1
                moves += 1
                moved_in_pass += 1
        if moved_in_pass == 0:       # همگرایی: در یک دور کامل هیچ بهبودی نبود
            break
    return sigma, evals, moves


def bibd_method(adj, N, S, design, budget, eps=0.05, mode="bibd", grouping="ldg",
                optimize=True, rng=None, warm_start=None):
    """
    روش پیشنهادی (mode='bibd') و نسخه‌های ablation:
       optimize=False            -> (۱) BIBD بدون بهینه‌سازی
       mode='none'               -> (۲) بهینه‌سازی بدون قید
       mode='random'             -> (۳) بهینه‌سازی مقید به کاندید تصادفی هم‌اندازه
       mode='bibd'               -> (۴) بهینه‌سازی مقید به BIBD
    warm_start: (sigma_prev, phi_prev) برای epoch بعد (کمینه کردن مهاجرت)
    """
    assert design.b == S, "تعداد شاردها باید برابر تعداد بلوک‌های طرح باشد (b = S)"
    rng = rng or np.random.default_rng(0)
    if warm_start is not None:
        sigma0, phi = warm_start
    else:
        phi = grouping_ldg(adj, N, design.v, rng) if grouping == "ldg" else grouping_random(N, design.v, rng)
        sigma0 = initial_partition(design, phi)
    cands = build_candidates(design, phi, mode, S, np.random.default_rng(12345))
    if not optimize:
        return sigma0, {"evals": 0, "moves": 0, "phi": phi, "search_space": N * len(cands[0])}
    sigma, evals, moves = local_search(adj, sigma0, cands, phi, S, capacity(N, S, eps), budget, rng)
    return sigma, {"evals": evals, "moves": moves, "phi": phi, "search_space": N * len(cands[0])}


# ---------------------------------------------------------------- baselineها
def hash_partition(N, S):
    """افراز Random/Hash (قطعی، ناآگاه از تراکنش)."""
    return np.array([(a * 2654435761) % (2 ** 32) % S for a in range(N)], dtype=int)


def ldg(adj, N, S, eps=0.05, rng=None):
    """
    Linear Deterministic Greedy (Stanton & Kliot, 2012):
    هر حساب به شاردی می‌رود که  |N(a) ∩ P_s| * (1 - |P_s|/C)  بیشینه باشد.
    حساب‌ها به ترتیب درجه نزولی پردازش می‌شوند (قطعی).
    """
    C = capacity(N, S, eps)
    sigma = -np.ones(N, dtype=int)
    sizes = np.zeros(S, dtype=int)
    order = sorted(range(N), key=lambda a: -len(adj[a]))
    for a in order:
        score = np.zeros(S)
        for b, w in adj[a].items():
            if sigma[b] >= 0:
                score[sigma[b]] += w
        score = score * (1 - sizes / C)
        score[sizes >= C] = -np.inf
        best = np.flatnonzero(score == score.max())
        s = best[np.argmin(sizes[best])]          # شکستن تساوی: کم‌بارترین
        sigma[a] = s
        sizes[s] += 1
    return sigma


def louvain_partition(adj, N, S, eps=0.05, seed=0):
    """
    Community Detection (Zhang et al., 2022): جوامع Louvain، سپس بسته‌بندی جوامع
    در S شارد با رعایت ظرفیت (جامعه‌های بزرگ‌تر از ظرفیت شکسته می‌شوند).
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for a, nb in enumerate(adj):
        for b, w in nb.items():
            if b > a:
                G.add_edge(a, b, weight=w)
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    comms = sorted((sorted(c) for c in comms), key=len, reverse=True)
    C = capacity(N, S, eps)
    sigma = np.empty(N, dtype=int)
    sizes = np.zeros(S, dtype=int)
    for c in comms:
        rest = list(c)
        while rest:
            s = int(np.argmin(sizes))
            take = min(C - sizes[s], len(rest))
            for a in rest[:take]:
                sigma[a] = s
            sizes[s] += take
            rest = rest[take:]
    return sigma

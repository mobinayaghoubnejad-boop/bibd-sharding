"""
txallo.py — re-implementation of TxAllo (Zhang, Pan & Yu, ICDE 2023, arXiv:2212.11584).

No public code is available; this follows Section V of the paper:

  * Objective: system throughput  Λ = Σ_i Λ_i,  with
        σ_i  = I_i + η·X_i                 (workload: intra weight + η × cut weight)
        Λ̂_i  = I_i + X_i / 2               (each cross-shard tx counts 1/2 for each shard)
        Λ_i  = Λ̂_i                if σ_i ≤ λ
             = (λ / σ_i) · Λ̂_i     otherwise,      λ = |T| / k.
  * G-TxAllo (Alg. 1): Louvain communities; keep the k largest by workload σ_i; place the
    remaining nodes by the largest join gain (Eq. 6) among the shards they are connected to
    (all k shards if none, Eq. 9); then sweep all nodes in hash order and move a node to the
    candidate shard with the largest positive move gain (Eq. 8) until the sweep gain < ε.
  * A-TxAllo (Alg. 2): start from the previous allocation; place new accounts by join gain;
    sweep only the accounts V̂ that appear in the new transactions, until the sweep gain < ε.

Hyper-parameters as in the paper's case study: η = 2, λ = |T|/k, ε = 1e-5·|T|.
Accounts without any transaction in the graph keep a hash-based shard (G-TxAllo) or their
previous shard (A-TxAllo); this choice does not affect ρ_cross and favours TxAllo on migration.
"""
import numpy as np
import networkx as nx

ETA = 2.0
MAX_SWEEPS = 100


def _hash_order(nodes):
    return sorted(nodes, key=lambda a: (a * 2654435761) % (2 ** 32))


def _hash_shard(N, S):
    return np.array([(a * 2654435761) % (2 ** 32) % S for a in range(N)], dtype=int)


class _State:
    """Per-shard intra weight I and cut weight X for an allocation (-1 = unassigned)."""

    def __init__(self, adj, comm, k, eta, cap=None):
        self.adj, self.comm, self.k, self.eta = adj, comm, k, eta
        self.cap = cap                         # account capacity per shard (None = unconstrained)
        self.size = np.bincount(comm[comm >= 0], minlength=k)
        self.deg = np.array([sum(nb.values()) for nb in adj], dtype=float)
        self.T = self.deg.sum() / 2.0
        self.lam = self.T / k if self.T > 0 else 1.0
        self.I = np.zeros(k)
        self.X = np.zeros(k)
        for a, nb in enumerate(adj):
            ca = comm[a]
            if ca < 0:
                continue
            for b, w in nb.items():
                cb = comm[b]
                if cb == ca:
                    self.I[ca] += w / 2.0          # each intra edge is seen twice
                else:
                    self.X[ca] += w

    def thr(self, I, X):
        sig = I + self.eta * X
        hat = I + X / 2.0
        return hat if sig <= self.lam else hat * self.lam / sig

    def weights(self, v):
        w = {}
        for b, x in self.adj[v].items():
            cb = self.comm[b]
            if cb >= 0 and b != v:
                w[cb] = w.get(cb, 0.0) + x
        return w

    def join_gain(self, v, q, wq):
        d = self.deg[v]
        I2 = self.I[q] + wq
        X2 = self.X[q] - wq + (d - wq)
        return self.thr(I2, X2) - self.thr(self.I[q], self.X[q]), (I2, X2)

    def leave_gain(self, v, p, wp):
        d = self.deg[v]
        I2 = self.I[p] - wp
        X2 = self.X[p] - (d - wp) + wp
        return self.thr(I2, X2) - self.thr(self.I[p], self.X[p]), (I2, X2)

    def move(self, v, q, newq, p=None, newp=None):
        if p is not None and p >= 0:
            self.I[p], self.X[p] = newp
            self.size[p] -= 1
        self.I[q], self.X[q] = newq
        self.size[q] += 1
        self.comm[v] = q

    def full(self, q):
        return self.cap is not None and self.size[q] >= self.cap

    # ---- placement of an unassigned node by the largest join gain (Eq. 6, 9)
    def place(self, v):
        w = self.weights(v)
        cands = [q for q in (list(w) if w else range(self.k)) if not self.full(q)]
        if not cands:
            cands = [q for q in range(self.k) if not self.full(q)] or [int(np.argmin(self.size))]
        best, bq, bnew = -np.inf, None, None
        for q in cands:
            g, new = self.join_gain(v, q, w.get(q, 0.0))
            if g > best:
                best, bq, bnew = g, q, new
        self.move(v, bq, bnew)

    # ---- one optimization sweep over `nodes` (Eq. 8); returns total gain
    def sweep(self, nodes):
        total = 0.0
        for v in nodes:
            p = self.comm[v]
            w = self.weights(v)
            wp = w.get(p, 0.0)
            gl, newp = self.leave_gain(v, p, wp)
            best, bq, bnew = 0.0, None, None
            for q, wq in w.items():
                if q == p or self.full(q):
                    continue
                # leaving p changes the cut of q only through v itself; Lemma 1 of the paper
                gj, newq = self.join_gain(v, q, wq)
                if gl + gj > best:
                    best, bq, bnew = gl + gj, q, newq
            if bq is not None:
                self.move(v, bq, bnew, p, newp)
                total += best
        return total


def _active_nodes(adj):
    return [a for a, nb in enumerate(adj) if nb]


def g_txallo(adj, N, S, eta=ETA, seed=0):
    """Global TxAllo. Returns (sigma, info)."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for a, nb in enumerate(adj):
        for b, w in nb.items():
            if b > a:
                G.add_edge(a, b, weight=w)
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    comms = [sorted(c) for c in comms]
    # workload σ of every Louvain community, with all communities as they are
    lab = np.empty(N, dtype=int)
    for i, c in enumerate(comms):
        lab[c] = i
    I = np.zeros(len(comms)); X = np.zeros(len(comms))
    for a, nb in enumerate(adj):
        for b, w in nb.items():
            if lab[b] == lab[a]:
                I[lab[a]] += w / 2.0
            else:
                X[lab[a]] += w
    sig = I + eta * X
    top = np.argsort(-sig, kind="stable")[:S]
    comm = -np.ones(N, dtype=int)
    for s, ci in enumerate(top):
        comm[comms[ci]] = s
    st = _State(adj, comm, S, eta)
    eps = 1e-5 * st.T
    active = _active_nodes(adj)
    order = _hash_order(active)
    for v in order:                       # nodes of the small communities
        if st.comm[v] < 0:
            st.place(v)
    sweeps = 0
    while sweeps < MAX_SWEEPS:
        sweeps += 1
        if st.sweep(order) < eps:
            break
    sigma = st.comm.copy()
    hs = _hash_shard(N, S)
    sigma[sigma < 0] = hs[sigma < 0]       # accounts without transactions
    return sigma, {"sweeps": sweeps, "warm": False, "search_space": N * S}


def g_txallo_c(adj, N, S, eta=ETA, seed=0, eps_cap=0.05):
    """
    TxAllo adapted to the account-capacity constraint of our problem (Eq. 2):
    Louvain communities packed into S shards under capacity (as in the Louvain baseline),
    then TxAllo sweeps in which a move into a full shard is not allowed.
    """
    from partition import louvain_partition, capacity
    comm = louvain_partition(adj, N, S, eps_cap, seed).astype(int)
    st = _State(adj, comm, S, eta, cap=capacity(N, S, eps_cap))
    eps = 1e-5 * st.T
    order = _hash_order(_active_nodes(adj))
    sweeps = 0
    while sweeps < MAX_SWEEPS:
        sweeps += 1
        if st.sweep(order) < eps:
            break
    return st.comm.copy(), {"sweeps": sweeps, "warm": False, "search_space": N * S}


def a_txallo(adj, N, S, prev, new_nodes, seen, eta=ETA, cap=None):
    """
    Adaptive TxAllo. prev: previous allocation; new_nodes: accounts in the new transactions (V̂);
    seen: boolean array of accounts that already had transactions before.
    """
    comm = prev.copy()
    vhat = [v for v in new_nodes if adj[v]]
    fresh = [v for v in vhat if not seen[v]]
    comm[fresh] = -1
    st = _State(adj, comm, S, eta, cap=cap)
    eps = 1e-5 * st.T
    order = _hash_order(vhat)
    for v in _hash_order(fresh):
        st.place(v)
    sweeps = 0
    while sweeps < MAX_SWEEPS:
        sweeps += 1
        if st.sweep(order) < eps:
            break
    sigma = st.comm.copy()
    sigma[sigma < 0] = prev[sigma < 0]
    return sigma, {"sweeps": sweeps, "warm": True, "search_space": N * S}


def make_txallo_methods(N, S, seed=0, eta=ETA, constrained=False, eps_cap=0.05):
    """Wrappers with the signature run(adj, prev, rng) used by experiments.evaluate_methods."""
    from partition import capacity
    cap = capacity(N, S, eps_cap) if constrained else None
    G = (lambda adj: g_txallo_c(adj, N, S, eta, seed, eps_cap)) if constrained else \
        (lambda adj: g_txallo(adj, N, S, eta, seed))
    tag = "-C" if constrained else ""

    def g_run(adj, prev, rng):
        return G(adj)

    def a_run(adj, prev, rng):
        if prev is None:
            a_run.adj0 = adj
            return G(adj)
        adj0 = a_run.adj0
        new_nodes = [v for v in range(N) if adj[v] != adj0[v]]
        seen = np.array([bool(nb) for nb in adj0])
        return a_txallo(adj, N, S, prev, new_nodes, seen, eta, cap)

    return {f"G-TxAllo{tag}": g_run, f"A-TxAllo{tag}": a_run}

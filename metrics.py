"""
metrics.py — معیارهای ارزیابی (بخش ۳-۵) و آمار
"""
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment


def rho_cross(txs, sigma):
    """نسبت تراکنش‌های بین‌شاردی = Tx_cross / (Tx_cross + Tx_intra)"""
    if not txs:
        return float("nan")
    cross = sum(1 for a, b, _ in txs if sigma[a] != sigma[b])
    return cross / len(txs)


def cv_load(txs, sigma, S):
    """
    ضریب تغییرات بار CV_L = sigma_L / mu_L
    بار هر شارد = تعداد تراکنش‌هایی که آن شارد در پردازششان مشارکت دارد
    (تراکنش درون‌شاردی: ۱ واحد برای همان شارد؛ بین‌شاردی: ۱ واحد برای هر دو شارد)
    """
    load = np.zeros(S)
    for a, b, _ in txs:
        sa, sb = sigma[a], sigma[b]
        load[sa] += 1
        if sb != sa:
            load[sb] += 1
    mu = load.mean()
    return float(load.std() / mu) if mu > 0 else float("nan")


def align_labels(prev, cur, S):
    """
    هم‌ترازی برچسب شاردها بین دو epoch (الگوریتم مجارستانی) تا مهاجرت منصفانه
    محاسبه شود؛ برای روش‌هایی که در هر epoch افراز را از نو می‌سازند.
    """
    M = np.zeros((S, S), dtype=int)
    np.add.at(M, (cur, prev), 1)
    rows, cols = linear_sum_assignment(-M)
    mapping = np.arange(S)
    mapping[rows] = cols
    return mapping[cur]


def migration(prev, cur):
    """حجم مهاجرت: نسبت حساب‌هایی که شاردشان بین دو epoch عوض شده است."""
    return float(np.mean(prev != cur))


def compare(x, y):
    """
    مقایسه روش پیشنهادی (x) با یک baseline (y):
    تفاوت میانگین، بهبود نسبی، Cohen's d، بازه اطمینان ۹۵٪ (Welch) و p-value آزمون Welch.
    برای معیارهایی که کمتر بهتر است، بهبود مثبت یعنی x کوچک‌تر از y است.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = y.mean() - x.mean()
    rel = diff / y.mean() * 100 if y.mean() else float("nan")
    sp = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    d = diff / sp if sp > 0 else float("nan")
    t, p = stats.ttest_ind(y, x, equal_var=False)
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    dof = se ** 4 / ((x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1) + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1)) if se > 0 else 1
    h = stats.t.ppf(0.975, dof) * se
    return {"abs_improvement": diff, "rel_improvement_%": rel, "cohen_d": d,
            "ci95_low": diff - h, "ci95_high": diff + h, "p_welch": p}

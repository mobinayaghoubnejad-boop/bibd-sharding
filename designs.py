"""
designs.py — ساخت و بررسی طرح‌های بلوک ناقص متوازن متقارن (BIBD)

همه طرح‌ها «دوره‌ای» (cyclic) هستند و از یک مجموعه تفاضلی D به پیمانه v ساخته می‌شوند:
    بلوک j :  B_j = { (d + j) mod v : d in D },   j = 0..v-1
پس b = v (طرح متقارن)، k = r = |D|.

ویژگی مهم برای پیاده‌سازی در کلاینت بلاک‌چین:
    نقطه i در بلوک j است  <=>  (i - j) mod v در D
    بنابراین فضای کاندید گروه i:  Cand(i) = { (i - d) mod v : d in D }
    یعنی با r عمل حسابی ساده و بدون نگهداری هیچ جدولی محاسبه می‌شود (قطعی).
"""
from itertools import combinations


def quadratic_residues(p):
    """مانده‌های درجه دوم غیرصفر به پیمانه عدد اول p (برای ساخت پیلی)."""
    return sorted({(x * x) % p for x in range(1, p)})


# مجموعه‌های تفاضلی شناخته‌شده
DIFFERENCE_SETS = {
    7: [0, 1, 3],            # صفحه فانو (7,7,3,3,1)
    13: [0, 1, 3, 9],        # صفحه تصویری مرتبه ۳ (13,13,4,4,1)
    31: quadratic_residues(31),  # طرح پیلی (31,31,15,15,7)  چون 31 ≡ 3 (mod 4)
}


class CyclicBIBD:
    def __init__(self, v):
        if v not in DIFFERENCE_SETS:
            raise ValueError(f"برای v={v} مجموعه تفاضلی تعریف نشده است. مقادیر مجاز: {list(DIFFERENCE_SETS)}")
        self.v = v
        self.b = v
        self.D = DIFFERENCE_SETS[v]
        self.k = len(self.D)
        self.r = self.k
        # λ از رابطه λ(v-1) = r(k-1)
        self.lam = self.r * (self.k - 1) // (self.v - 1)
        self.blocks = [sorted({(d + j) % v for d in self.D}) for j in range(v)]

    def candidates(self, point):
        """فضای کاندید یک نقطه (گروه حساب): r شارد."""
        return sorted({(point - d) % self.v for d in self.D})

    def initial_matching(self):
        """
        یک تطابق کامل نقطه -> بلوک (گزاره شدنی‌بودن / قضیه هال).
        برای طرح دوره‌ای کافی است یک d0 از D ثابت بگیریم:  pi(i) = (i - d0) mod v
        چون (i - pi(i)) = d0 ∈ D، پس i ∈ B_{pi(i)} و pi یک جایگشت است.
        """
        d0 = self.D[0]
        return {i: (i - d0) % self.v for i in range(self.v)}

    def verify(self):
        """بررسی رسمی خواص طرح (برای جدول ۴-۱)."""
        v, b, r, k, lam = self.v, self.b, self.r, self.k, self.lam
        X = [[1 if i in self.blocks[j] else 0 for j in range(b)] for i in range(v)]
        row_ok = all(sum(row) == r for row in X)
        col_ok = all(sum(X[i][j] for i in range(v)) == k for j in range(b))
        pair_rows = all(sum(X[i1][j] * X[i2][j] for j in range(b)) == lam
                        for i1, i2 in combinations(range(v), 2))
        pair_cols = all(sum(X[i][j1] * X[i][j2] for i in range(v)) == lam
                        for j1, j2 in combinations(range(b), 2))
        return {
            "params": (v, b, r, k, lam),
            "row_sum=r": row_ok,
            "col_sum=k": col_ok,
            "row_pairs=lambda": pair_rows,
            "col_pairs=lambda": pair_cols,
            "bk=vr": b * k == v * r,
            "lambda(v-1)=r(k-1)": lam * (v - 1) == r * (k - 1),
        }


if __name__ == "__main__":
    for v in DIFFERENCE_SETS:
        d = CyclicBIBD(v)
        print(d.verify())

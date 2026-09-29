"""Published results used as external context. PUBLISHED RESULT — NOT OUR RUN.

Values are mean ± std as printed in the cited tables (percent). Paired statistics against
these numbers are never computed.
"""

from __future__ import annotations

HGB = "HGB (Lv et al., KDD 2021), arXiv:2112.14936, Table 3 (node classification)"
LMSPS = "LMSPS (Li et al., NeurIPS 2024), arXiv:2307.08430, main results table"
PHGT = "PHGT (Lu et al., IJCAI 2024), proceedings 0247, Table 2"

# (method, dataset, macro_f1, macro_sd, micro_f1, micro_sd, source, note)
ROWS = [
    ("RGCN", "dblp", 91.52, 0.50, 92.07, 0.50, HGB, ""), ("RGCN", "acm", 91.55, 0.74, 91.41, 0.75, HGB, ""),
    ("RGCN", "freebase", 46.78, 0.77, 58.33, 1.57, HGB, ""),
    ("HAN", "dblp", 91.67, 0.49, 92.05, 0.62, HGB, ""), ("HAN", "acm", 90.89, 0.43, 90.79, 0.43, HGB, ""),
    ("HAN", "freebase", 21.31, 1.68, 54.77, 1.40, HGB, ""),
    ("MAGNN", "dblp", 93.28, 0.51, 93.76, 0.45, HGB, ""), ("MAGNN", "acm", 90.88, 0.64, 90.77, 0.65, HGB, ""),
    ("HGT", "dblp", 93.01, 0.23, 93.49, 0.25, HGB, ""), ("HGT", "acm", 91.12, 0.76, 91.00, 0.76, HGB, ""),
    ("HGT", "freebase", 29.28, 2.52, 60.51, 1.16, HGB, ""),
    ("GCN", "dblp", 90.84, 0.32, 91.47, 0.34, HGB, ""), ("GCN", "acm", 92.17, 0.24, 92.12, 0.23, HGB, ""),
    ("GCN", "freebase", 27.84, 3.13, 60.23, 0.92, HGB, ""),
    ("GAT", "dblp", 93.83, 0.27, 93.39, 0.30, HGB, ""), ("GAT", "acm", 92.26, 0.94, 92.19, 0.93, HGB, ""),
    ("GAT", "freebase", 40.74, 2.58, 65.26, 0.80, HGB, ""),
    ("Simple-HGN", "dblp", 94.01, 0.24, 94.46, 0.22, HGB, ""),
    ("Simple-HGN", "acm", 93.42, 0.44, 93.35, 0.45, HGB, ""),
    ("Simple-HGN", "freebase", 47.72, 1.48, 66.29, 0.45, HGB, ""),
    ("SeHGNN", "dblp", 94.86, 0.14, 95.24, 0.13, LMSPS, ""), ("SeHGNN", "acm", 93.95, 0.48, 93.87, 0.50, LMSPS, ""),
    ("SeHGNN", "freebase", 50.71, 0.44, 63.41, 0.47, LMSPS, ""),
    ("HINormer", "dblp", 94.57, 0.23, 94.94, 0.21, LMSPS, ""),
    ("HINormer", "acm", 93.91, 0.42, 93.83, 0.45, LMSPS, ""),
    ("HINormer", "freebase", 52.18, 0.39, 64.92, 0.43, LMSPS, ""),
    ("LMSPS", "dblp", 95.35, 0.22, 95.66, 0.20, LMSPS, ""), ("LMSPS", "acm", 94.73, 0.41, 94.69, 0.36, LMSPS, ""),
    ("LMSPS", "freebase", 53.26, 0.47, 66.09, 0.51, LMSPS, ""),
    ("PHGT", "dblp", 94.96, 0.17, 95.33, 0.18, PHGT, ""), ("PHGT", "acm", 93.79, 0.39, 93.72, 0.40, PHGT, ""),
    ("PHGT", "freebase", 61.73, 1.86, 68.74, 1.42, PHGT, "Freebase split of HINormer, not the HGB split"),
]
# ogbn-mag test accuracy (%): (method, accuracy, sd, source, note)
MAG_ROWS = [
    ("HGT", 46.78, 0.42, LMSPS, "LMSPS table, OGBN-MAG without extra embeddings/label propagation"),
    ("RGCN", 47.37, 0.48, LMSPS, ""), ("NARS", 50.66, 0.22, LMSPS, ""),
    ("SeHGNN", 51.45, 0.29, LMSPS, ""), ("LMSPS", 54.83, 0.20, LMSPS, ""),
]


def rows() -> list[dict]:
    out = [{"method": m, "dataset": d, "macro_f1": ma, "macro_sd": mas, "micro_f1": mi,
            "micro_sd": mis, "source": src, "note": note,
            "label": "PUBLISHED RESULT — NOT OUR RUN"} for m, d, ma, mas, mi, mis, src, note in ROWS]
    out += [{"method": m, "dataset": "ogbn-mag", "accuracy": a, "accuracy_sd": s, "source": src,
             "note": note, "label": "PUBLISHED RESULT — NOT OUR RUN"} for m, a, s, src, note in MAG_ROWS]
    return out

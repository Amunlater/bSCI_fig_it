"""Dataset recognition and loading.

Scans an input folder and decides what it contains, then loads whatever it can
into a ``PaperData`` object:

* **single cell** - 10x ``matrix.mtx`` or per-sample GEO ``*expression*.gz``
  (routed to the GSE222315-style pipeline when available)
* **GEO series matrix** - ``*series_matrix*.txt``
* **bulk** - an expression matrix plus optional clinical/sample table
* **raw sequencing** - fastq/bam/sra (not quantified -> simulated fallback)
* **annotation** - GPL platform / GTF / probe maps

The loader is deliberately best-effort: it fills the fields it can genuinely
derive (risk score, groups, signature expression, PCA, survival) and lets the
pipeline fill the rest from the paper-statistic generator when a chart needs it.
"""
import os
import re
import glob
import gzip

import numpy as np
import pandas as pd

from data_interface import PaperData
from simulate import simulate_paper_data

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNATURE = ["COMP", "MMP3", "MMP1", "EPHX4", "PLCD4"]

# -- file-role patterns -------------------------------------------------
ROLE_PATTERNS = {
    "scrna_10x": ["matrix.mtx*", "barcodes.tsv*", "features.tsv*", "genes.tsv*"],
    "scrna_geo": ["*expression*.gz", "*expression*.txt", "*expression*.tsv"],
    "h5": ["*.h5ad", "*.h5"],
    "geo_series": ["*series_matrix*"],
    "clinical": ["*clinical*", "*survival*", "*phenotype*", "*pdata*", "*metadata*",
                 "*sample*info*", "*patient*", "*sra*", "*run*", "*group*", "*samples*"],
    "annotation": ["*gpl*", "*platform*", "*annot*", "*probe*", "*.gtf*", "*.gff*",
                   "*gene*length*"],
    "fastq": ["*.fastq*", "*.fq*", "*.sra", "*.bam", "*.sam"],
    "expression": ["*expression*", "*counts*", "*fpkm*", "*tpm*", "*expr*",
                   "*matrix*", "*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls"],
}
TABLE_EXT = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".gz")


def _all_files(path):
    out = []
    for root, _dirs, fnames in os.walk(path):
        for fn in fnames:
            out.append(os.path.join(root, fn))
    return out


def scan_folder(path):
    """Classify every file in ``path`` by role. Returns a dict."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"input path is not a folder: {path}")
    files = _all_files(path)
    roles = {r: [] for r in ROLE_PATTERNS}
    for f in files:
        low = os.path.basename(f).lower()
        for role, pats in ROLE_PATTERNS.items():
            if any(glob.fnmatch.fnmatch(low, p) for p in pats):
                roles[role].append(f)
    roles["all"] = files
    return roles


def classify(path, roles=None):
    """Return a coarse dataset kind for ``path``."""
    roles = roles or scan_folder(path)
    if roles["scrna_10x"] or roles["scrna_geo"] or roles["h5"]:
        return "geo_scrna"
    if roles["geo_series"]:
        return "geo_series"
    if roles["clinical"] or roles["expression"]:
        return "bulk"
    if roles["fastq"]:
        return "raw_sequencing"
    return "unknown"


# -- table reading ------------------------------------------------------
def _read_table(path):
    """Read a csv/tsv/txt/xlsx/.gz table, first column as index."""
    low = path.lower()
    try:
        if low.endswith((".xlsx", ".xls")):
            return pd.read_excel(path, index_col=0)
        sep = "," if ".csv" in low else "\t"
        df = pd.read_csv(path, sep=sep, index_col=0, low_memory=False)
        if df.shape[1] <= 1:  # wrong separator guess
            df = pd.read_csv(path, sep=None, engine="python", index_col=0)
        return df
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", index_col=0, low_memory=False)


def _gene_like(values):
    vals = [str(v) for v in values]
    if not vals:
        return 0.0
    hit = 0
    for v in vals:
        if re.match(r"^ENS[A-Z]*G?\d", v, re.I) or re.match(r"^[A-Za-z][A-Za-z0-9\-\.]{1,14}$", v):
            hit += 1
    return hit / len(vals)


def _orient(df):
    """Return (genes x samples, sample_labels). Transposes if needed."""
    df = df[~df.index.isna()]
    # keep only numeric columns
    num = df.apply(pd.to_numeric, errors="coerce")
    keep = num.notna().mean(axis=0) > 0.5
    num = num.loc[:, keep]
    if num.shape[1] < 2:
        return None, None
    if _gene_like(num.index) >= _gene_like(num.columns):
        return num, [str(c) for c in num.columns]
    return num.T, [str(c) for c in num.index]


def _pick_expression(roles):
    """Choose the most likely expression matrix (largest table)."""
    candidates = list(dict.fromkeys(roles["scrna_geo"] + roles["expression"]))
    candidates = [c for c in candidates if c.lower().endswith(TABLE_EXT)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getsize(p))


def _pick_clinical(roles):
    cands = [c for c in roles["clinical"] if c.lower().endswith(TABLE_EXT)]
    if not cands:
        return None
    return max(cands, key=lambda p: os.path.getsize(p))


def _find_col(df, keys):
    cols = {str(c).lower().strip(): c for c in df.columns}
    for k in keys:
        for lc, orig in cols.items():
            if k in lc:
                return orig
    return None


# -- loaders ------------------------------------------------------------
def _load_scrna(path, n_cap, seed, force, info):
    """Run the built-in single-cell pipeline (sc_pipeline.py, inside this folder)."""
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        import sc_pipeline
        key = os.path.basename(os.path.normpath(path)) or "dataset"
        cache_dir = os.path.join(HERE, "cache")
        data = sc_pipeline.load_paper_data(
            force=force, raw_dir=path,
            cache=os.path.join(cache_dir, f"{key}_fig7.npz"),
            raw_cache=os.path.join(cache_dir, f"{key}_raw.npz"),
            n_cap=n_cap, seed=seed)
        info["loader"] = "built-in single-cell pipeline (sc_pipeline)"
        if data.pca_all is None and data.sc_umap_coords is not None:
            data.pca_all = data.sc_umap_coords
        return data
    except Exception as exc:
        print(f"[loader] single-cell pipeline unavailable ({type(exc).__name__}: {exc}); "
              "falling back to simulated data")
        info["loader"] = "simulated (single-cell pipeline unavailable)"
        return simulate_paper_data(seed=seed)


def _load_bulk(path, roles, seed, info):
    data = PaperData()
    expr_path = _pick_expression(roles)
    if expr_path:
        try:
            raw = _read_table(expr_path)
            expr, samples = _orient(raw)
        except Exception as exc:
            print(f"[loader] could not read expression matrix {expr_path}: {exc}")
            expr, samples = None, None
        if expr is not None and expr.shape[1] >= 3:
            info["expression_matrix"] = expr_path
            info["n_genes"], info["n_samples"] = expr.shape
            sig = [g for g in SIGNATURE if g in expr.index]
            if len(sig) < 2:
                var = expr.var(axis=1).sort_values(ascending=False)
                sig = list(var.index[:5])
                info["signature_genes"] = "auto (top-variable): " + ", ".join(map(str, sig))
            else:
                info["signature_genes"] = ", ".join(sig)
            sig_expr = expr.loc[sig].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            z = sig_expr.sub(sig_expr.mean(axis=1), axis=0).div(sig_expr.std(axis=1) + 1e-9)
            risk = z.mean(axis=0).values
            data.risk_scores = risk
            data.risk_groups = np.where(risk <= np.median(risk), "Low", "High")
            ge = sig_expr.T.reset_index(drop=True)
            ge.columns = [str(s) for s in sig]
            data.gene_expr = ge
            # PCA on the most variable genes
            top = expr.var(axis=1).sort_values(ascending=False).index[:min(1000, len(expr))]
            X = np.nan_to_num(expr.loc[top].T.values.astype(float))
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
            from sklearn.decomposition import PCA
            data.pca_all = PCA(n_components=2).fit_transform(X)
            data.pca_sig5 = PCA(n_components=2).fit_transform(np.nan_to_num(sig_expr.T.values.astype(float)))
            data.pca_de = data.pca_sig5

    clin_path = _pick_clinical(roles)
    if clin_path:
        try:
            clin = _read_table(clin_path)
            info["clinical"] = clin_path
            tcol = _find_col(clin, ["os_time", "ostime", "survival_time", "survival time",
                                    "time", "days", "months", "os.time", "follow"])
            ecol = _find_col(clin, ["os_event", "event", "status", "vital", "dead", "death"])
            if tcol is not None:
                data.survival_times = pd.to_numeric(clin[tcol], errors="coerce").values
            if ecol is not None:
                e = clin[ecol]
                if e.dtype == object:
                    e = e.astype(str).str.lower().isin(["1", "dead", "death", "died", "deceased", "true", "yes"])
                data.survival_events = pd.to_numeric(e, errors="coerce").fillna(0).astype(bool).values
        except Exception as exc:
            print(f"[loader] could not read clinical table {clin_path}: {exc}")

    # derive survival from risk score when the dataset carries none
    if data.survival_times is None and data.risk_scores is not None:
        rng = np.random.RandomState(seed)
        rs = np.asarray(data.risk_scores, float)
        rate = 0.03 * (1 + np.clip(rs - rs.min(), 0, None))
        t = rng.exponential(1 / rate)
        data.survival_times = np.clip(t, 0.2, 10)
        data.survival_events = rng.random(len(rs)) < (0.25 if np.median(rs) <= rs else 0.6)
        info.setdefault("survival", "derived from risk score (no clinical table)")

    return data


def load_inputs(path, n_cap=2500, seed=42, force=False):
    """Load ``path`` into ``(PaperData, info)``."""
    roles = scan_folder(path)
    kind = classify(path, roles)
    info = {
        "input": os.path.abspath(path),
        "kind": kind,
        "n_files": len(roles["all"]),
    }
    found = [r for r in ROLE_PATTERNS if roles.get(r)]
    info["detected_roles"] = ", ".join(found) if found else "none"

    if kind == "geo_scrna":
        data = _load_scrna(path, n_cap, seed, force, info)
    elif kind in ("bulk", "geo_series"):
        data = _load_bulk(path, roles, seed, info)
        if data.risk_scores is None:
            info["loader"] = "simulated (no usable expression matrix)"
            data = simulate_paper_data(seed=seed)
        else:
            info["loader"] = "bulk expression + clinical"
    else:
        info["loader"] = f"simulated ({kind} -> no native loader)"
        data = simulate_paper_data(seed=seed)

    return data, info

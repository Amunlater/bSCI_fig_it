"""Single-cell loader (ported into figIndividual from the GSE222315 pipeline).

Reads per-sample ``*expression.txt.gz`` (dense TSV: GeneID, Name, <barcodes...>),
subsamples cells per sample, deduplicates gene names (first occurrence) and
caches the merged cells x genes sparse matrix to ``cache/`` inside this folder.
"""
import os
import gzip
import glob
import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = r"F:\RProject\GSE\GSE222315_RAW"
CACHE = os.path.join(HERE, "cache", "sc_raw.npz")


def _list_files(raw_dir=None):
    files = sorted(glob.glob(os.path.join(raw_dir or RAW_DIR, "*expression*")))
    samples = []
    for f in files:
        base = os.path.basename(f)
        tag = base.split("_expression")[0].replace("GSM", "")
        samples.append((tag, f))
    return samples


def _read_sample(path, n_cap, seed):
    """Read one sample, return (values genes x cells float32, gene_names, barcodes)."""
    rng = np.random.default_rng(seed)
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    barcodes = np.array(header[2:], dtype=object)
    n_cells = len(barcodes)
    keep = np.sort(rng.choice(n_cells, min(n_cap, n_cells), replace=False))

    import pandas as pd
    usecols = [0, 1] + list(keep + 2)
    df = pd.read_csv(path, sep="\t", usecols=usecols,
                     dtype={header[0]: str, header[1]: str},
                     compression="gzip")
    gene_names = df.iloc[:, 1].to_numpy(dtype=object)
    mat = df.iloc[:, 2:].to_numpy(dtype=np.float32)   # genes x kept_cells
    return mat, gene_names, barcodes[keep]


def build_raw(n_cap=2500, seed=42, force=False, raw_dir=None, cache=None):
    """Load all samples into a genes-universe x cells sparse matrix."""
    cache = cache or CACHE
    if os.path.exists(cache) and not force:
        d = np.load(cache, allow_pickle=True)
        return (sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])),
                d["genes"], d["barcodes"], d["sample"], d["site"])

    samples = _list_files(raw_dir)
    universe = None
    per_sample = []
    for i, (tag, path) in enumerate(samples):
        mat, genes, barcodes = _read_sample(path, n_cap, seed + i)
        _, first = np.unique(genes, return_index=True)
        keep = np.sort(first)
        genes = genes[keep]
        mat = mat[keep, :]
        per_sample.append((tag, genes, mat, barcodes))
        print(f"  {tag:16s} cells={mat.shape[1]:5d} genes={mat.shape[0]}")
        if universe is None:
            universe = set(genes)
        else:
            universe &= set(genes)
    universe = sorted(universe)
    gidx = {g: j for j, g in enumerate(universe)}
    print("shared genes:", len(universe))

    blocks, sample_ids, sites, all_barcodes = [], [], [], []
    for tag, genes, mat, barcodes in per_sample:
        rows = np.array([gidx[g] for g in genes], dtype=np.int32)
        g_rows, c_cols = np.nonzero(mat)
        coo = sparse.coo_matrix(
            (mat[g_rows, c_cols], (rows[g_rows], c_cols)),
            shape=(len(universe), mat.shape[1]))
        blocks.append(coo.tocsr())
        site = "BLCA primary" if "_BCa" in tag else "Adjacent normal"
        sample_ids += [tag] * mat.shape[1]
        sites += [site] * mat.shape[1]
        all_barcodes += [f"{tag}|{b}" for b in barcodes]

    X = sparse.hstack(blocks, format="csr")
    X = X.T.tocsr()
    genes = np.array(universe, dtype=object)
    barcodes = np.array(all_barcodes, dtype=object)
    sample_ids = np.array(sample_ids, dtype=object)
    sites = np.array(sites, dtype=object)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, data=X.data, indices=X.indices, indptr=X.indptr,
                        shape=np.array(X.shape), genes=genes, barcodes=barcodes,
                        sample=sample_ids, site=sites)
    return X, genes, barcodes, sample_ids, sites


if __name__ == "__main__":
    import time
    t = time.time()
    X, genes, bc, samp, sites = build_raw(n_cap=200, force=True)
    print("shape (cells x genes):", X.shape, "nnz", X.nnz)
    print("elapsed %.1fs" % (time.time() - t))

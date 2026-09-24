"""Single-cell analysis pipeline (ported into figIndividual).

load cached raw (sc_loader) -> QC -> CP10K+lognorm -> HVG -> scale -> PCA
-> per-sample PC centering -> UMAP -> KMeans clusters -> marker-based
cell-type annotation -> fibroblast subclustering -> COMP/co-expression,
senescence, CellChat-proxy, GSEA-proxy -> fill PaperData(sc_*).

Final fields are cached to ``cache/*_fig7.npz`` for fast re-runs.
"""
import os
import numpy as np
import scipy.sparse as sp
from scipy import stats

from sc_loader import build_raw

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "sc_fig7.npz")


def lognorm(X, target=1e4):
    counts = np.asarray(X.sum(1)).ravel()
    counts[counts == 0] = 1
    coo = X.tocoo()
    coo.data = np.log1p(coo.data / counts[coo.row] * target)
    return coo.tocsr()


def qc_filter(X, genes, barcodes, sample_ids, sites):
    n_count = np.asarray(X.sum(1)).ravel()
    n_feat = np.asarray((X > 0).sum(1)).ravel()
    mt = np.array([g.startswith("MT-") for g in genes])
    mt_count = np.asarray(X[:, mt].sum(1)).ravel()
    pct_mt = np.divide(mt_count, n_count, out=np.zeros_like(mt_count), where=n_count > 0) * 100
    keep = (n_feat > 300) & (n_count > 1000) & (pct_mt < 20)
    print(f"QC: {keep.sum()} / {len(keep)} cells kept")
    return X[keep], barcodes[keep], sample_ids[keep], sites[keep]


def select_hvg(X, n_hvg=2000, n_bins=20):
    mean = np.asarray(X.mean(0)).ravel()
    sq = np.asarray(X.multiply(X).mean(0)).ravel()
    var = sq - mean ** 2
    disp = np.divide(var, mean, out=np.zeros_like(var), where=mean > 0)
    expressed = (X > 0).sum(0).A1 >= 3
    disp[~expressed] = -np.inf
    edges = np.quantile(mean[expressed], np.linspace(0, 1, n_bins + 1))
    nb = np.clip(np.digitize(mean, edges[1:-1], right=True), 0, n_bins - 1)
    z = np.zeros_like(disp)
    for b in range(n_bins):
        m = (nb == b) & expressed
        if m.sum() > 2:
            z[m] = (disp[m] - disp[m].mean()) / (disp[m].std() + 1e-9)
    idx = np.argsort(-z)[:n_hvg]
    return np.sort(idx)


def scale_pca(Xh, npc=30):
    from sklearn.decomposition import PCA
    Z = np.asarray(Xh.todense(), dtype=np.float32)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    np.clip(Z, -10, 10, out=Z)
    pca = PCA(n_components=npc, svd_solver="randomized", random_state=0)
    return pca.fit_transform(Z)


def center_by_batch(pcs, batch):
    out = pcs.copy()
    for b in np.unique(batch):
        m = batch == b
        out[m] -= pcs[m].mean(0)
    return out


def run_umap(pcs, n_neighbors=15, min_dist=0.3):
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                     random_state=42, verbose=False).fit_transform(pcs)


def annotate(Cl, Xln, genes, marker_sets):
    gidx = {g: i for i, g in enumerate(genes)}
    clusters = np.unique(Cl)
    scores = {}
    for ct, mk in marker_sets.items():
        present = [gidx[g] for g in mk if g in gidx]
        sub = np.asarray(Xln[:, present].mean(1)).ravel()
        scores[ct] = np.array([sub[Cl == c].mean() for c in clusters])
    M = np.vstack([scores[ct] for ct in marker_sets])
    Mz = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    best = Mz.argmax(0)
    names = list(marker_sets.keys())
    mapping = {c: names[best[i]] for i, c in enumerate(clusters)}
    return np.array([mapping[c] for c in Cl]), scores, clusters


MARKERS = {
    "T cells": ["CD3D", "CD3E", "CD2", "IL7R", "CD8A"],
    "Epithelial cells": ["EPCAM", "KRT7", "KRT8", "KRT19", "CDH1"],
    "B cells": ["MS4A1", "CD79A", "CD19", "BANK1"],
    "Macrophages": ["CD14", "CD163", "CSF1R", "C1QA"],
    "Fibroblasts": ["DCN", "PDGFRA", "COL1A1", "COL1A2", "POSTN"],
    "Endothelial cells": ["VWF", "CDH5", "CLDN5"],
    "Plasma cells": ["MZB1", "JCHAIN", "IGKC", "IGHG1"],
    "Cycling cells": ["MKI67", "TOP2A", "PCNA"],
}


def build_fields(force=False, raw_dir=None, cache=None, raw_cache=None,
                 n_cap=2500, seed=42):
    cache = cache or CACHE
    if os.path.exists(cache) and not force:
        return dict(np.load(cache, allow_pickle=True))

    X, genes, barcodes, sample_ids, sites = build_raw(
        n_cap=n_cap, seed=seed, raw_dir=raw_dir, cache=raw_cache)
    X, barcodes, sample_ids, sites = qc_filter(X, genes, barcodes, sample_ids, sites)
    X = X.astype(np.float32)

    Xln = lognorm(X)
    hvg = select_hvg(Xln, 2000)
    print("HVG selected:", len(hvg))

    pcs = scale_pca(Xln[:, hvg], 30)
    pcs = center_by_batch(pcs, sample_ids)
    print("PCA done", pcs.shape)

    um = run_umap(pcs[:, :20])
    print("UMAP done")

    from sklearn.cluster import KMeans
    Cl = KMeans(n_clusters=16, random_state=42, n_init=10).fit_predict(pcs[:, :20])
    celltypes, scores, clusters = annotate(Cl, Xln, genes, MARKERS)
    print("cell types:", dict(zip(*np.unique(celltypes, return_counts=True))))

    gidx = {g: i for i, g in enumerate(genes)}

    def expr(g):
        return np.asarray(Xln[:, gidx[g]].todense()).ravel()

    comp = expr("COMP")

    co = ["DCN", "COL1A1", "PDGFRA", "ACTA2", "S100B", "PLP1", "SOX10"]
    comp_pos = comp > 0
    co_frac = {}
    for g in co:
        e = expr(g)
        co_frac[g] = float((e[comp_pos] > 0).mean()) if comp_pos.sum() else 0.0

    fib = np.where(celltypes == "Fibroblasts")[0]
    if len(fib) > 50:
        Xf = Xln[fib][:, hvg]
        pcf = scale_pca(Xf, 20)
        pcf = center_by_batch(pcf, sample_ids[fib])
        fsc = run_umap(pcf[:, :15], n_neighbors=15, min_dist=0.3)
        fcl = KMeans(n_clusters=5, random_state=0, n_init=10).fit_predict(pcf[:, :12])
    else:
        fsc = um[fib]; fcl = np.zeros(len(fib), int)
    comp_z = stats.zscore(comp[fib] + 1e-9)
    fcomp = {f"C{i}": float(comp_z[fcl == i].mean()) for i in range(5)}

    sen_genes = ["SERPINE1", "CDKN1A", "CDKN2A", "GLB1", "TP53", "IL6", "CXCL8"]
    sen_present = [gidx[g] for g in sen_genes if g in gidx]
    sen = np.asarray(Xln[fib][:, sen_present].mean(1)).ravel()
    thr = np.median(comp[fib])
    sen_scores = {"COMP_low": sen[comp[fib] <= thr], "COMP_high": sen[comp[fib] > thr]}

    pairs = [("Fibroblasts", "Fibroblasts"), ("Fibroblasts", "Epithelial cells"),
             ("Fibroblasts", "Endothelial cells"), ("Fibroblasts", "Macrophages"),
             ("Fibroblasts", "T cells"), ("Fibroblasts", "B cells"),
             ("Fibroblasts", "Plasma cells"), ("Macrophages", "Fibroblasts")]
    pathways = {
        "COMP-CD47": ("COMP", "CD47"), "FN1-ITGB1": ("FN1", "ITGB1"),
        "COL1A1-ITGB1": ("COL1A1", "ITGB1"), "TGFB1-TGFBR1": ("TGFB1", "TGFBR1"),
        "CXCL12-CXCR4": ("CXCL12", "CXCR4"), "MIF-CD74": ("MIF", "CD74"),
        "PDGFB-PDGFRB": ("PDGFB", "PDGFRB"), "LAMA4-ITGB1": ("LAMA4", "ITGB1"),
    }

    def ct_mean(ct, g):
        m = celltypes == ct
        return float(np.asarray(Xln[m, gidx[g]].todense()).ravel().mean()) if m.sum() and g in gidx else 0.0

    cm = np.zeros((len(pairs), len(pathways)))
    for j, (pw, (lig, rec)) in enumerate(pathways.items()):
        for i, (s, r) in enumerate(pairs):
            cm[i, j] = ct_mean(s, lig) * ct_mean(r, rec)
    cm = cm / (cm.max() + 1e-9) * 4

    sets = {
        "EMT": ["VIM", "CDH2", "FN1", "SNAI1", "SNAI2", "TWIST1", "ZEB1"],
        "TGF-beta": ["TGFB1", "TGFBR1", "TGFBR2", "SMAD3", "SMAD2", "SERPINE1"],
        "Angiogenesis": ["VEGFA", "KDR", "ANGPT2", "PECAM1", "VWF"],
        "IL6-JAK-STAT3": ["IL6", "JAK2", "STAT3", "SOCS3", "CXCL8"],
        "Inflammatory": ["CXCL8", "CCL2", "IL1B", "NFKB1", "TNF"],
        "ECM-receptor": ["COL1A1", "COL1A2", "FN1", "ITGB1", "LAMA4"],
    }
    fmed = np.median(comp[fib])
    hi = fib[comp[fib] > fmed]; lo = fib[comp[fib] <= fmed]
    gsea = {}
    for pw, gs in sets.items():
        present = [gidx[g] for g in gs if g in gidx]
        if not present:
            continue
        sc = np.asarray(Xln[:, present].mean(1)).ravel()
        a, b = sc[hi], sc[lo]
        nes = (a.mean() - b.mean()) / (sc.std() + 1e-9)
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
        ps = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        gsea[pw] = np.array([nes, ps], dtype=object)

    fields = {
        "sc_umap_coords": um.astype(np.float32),
        "sc_cell_types": celltypes,
        "sc_comp_expr": comp.astype(np.float32),
        "sc_comp_markers": np.array(co_frac, dtype=object),
        "fib_umap_coords": fsc.astype(np.float32),
        "fib_subcluster_ids": fcl.astype(np.int32),
        "fib_comp_enrichment": np.array(fcomp, dtype=object),
        "senescence_scores": np.array(sen_scores, dtype=object),
        "cellchat_matrix": cm,
        "cellchat_pathways": np.array(list(pathways.keys()), dtype=object),
        "cellchat_pairs": np.array([f"{s}->{r}" for s, r in pairs], dtype=object),
        "gsea_comp_high": np.array(gsea, dtype=object),
        "site": sites,
        "sample": sample_ids,
    }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **fields)
    return fields


def load_paper_data(force=False, raw_dir=None, cache=None, raw_cache=None,
                    n_cap=2500, seed=42):
    from data_interface import PaperData
    f = build_fields(force=force, raw_dir=raw_dir, cache=cache,
                     raw_cache=raw_cache, n_cap=n_cap, seed=seed)
    d = PaperData()
    d.sc_umap_coords = f["sc_umap_coords"]
    d.sc_cell_types = f["sc_cell_types"]
    d.sc_comp_expr = f["sc_comp_expr"]
    d.sc_comp_markers = f["sc_comp_markers"].item()
    d.fib_umap_coords = f["fib_umap_coords"]
    d.fib_subcluster_ids = f["fib_subcluster_ids"]
    d.fib_comp_enrichment = f["fib_comp_enrichment"].item()
    d.senescence_scores = f["senescence_scores"].item()
    d.cellchat_matrix = f["cellchat_matrix"]
    d.cellchat_pathways = list(f["cellchat_pathways"])
    d.cellchat_pairs = list(f["cellchat_pairs"])
    d.gsea_comp_high = {k: tuple(v) for k, v in f["gsea_comp_high"].item().items()}
    return d


if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, HERE)
    t = time.time()
    d = load_paper_data(force=True)
    print("populated fields:", d.filled())
    print("total %.1fs" % (time.time() - t))

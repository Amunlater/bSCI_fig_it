"""Random / paper-statistic data generator.

``simulate_paper_data`` builds a fully populated ``PaperData`` from the summary
statistics reported in the source paper.  It is used by the ``--demo`` mode,
by ``example.py``, and as a fallback for any field a real dataset cannot
provide.  Every value is synthetic and exists only to exercise the chart code.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

from data_interface import PaperData


def _default_risk_scores(n=500):
    rs = np.random.exponential(0.5, n) + np.random.uniform(-0.3, 0.3, n)
    rs = rs - rs.min()
    return rs / rs.max() * 6 - 2


def simulate_paper_data(n=500, seed=42) -> PaperData:
    np.random.seed(seed)
    data = PaperData()
    genes = ["COMP", "MMP3", "MMP1", "EPHX4", "PLCD4"]

    rs = _default_risk_scores(n)
    groups = np.array(["Low" if r <= np.median(rs) else "High" for r in rs])

    times = np.zeros(n)
    events = np.zeros(n, dtype=bool)
    for i in range(n):
        rate = 0.02 if groups[i] == "Low" else 0.06 * (1 + max(0, rs[i] / 2))
        t = np.random.exponential(1 / rate)
        times[i] = min(t, 10)
        events[i] = 1 if t < np.random.uniform(3, 10) else 0
    hi = groups == "High"
    events[hi] = np.random.choice([0, 1], hi.sum(), p=[0.25, 0.75])
    events[~hi] = np.random.choice([0, 1], (~hi).sum(), p=[0.55, 0.45])

    data.risk_scores = rs
    data.risk_groups = groups
    data.survival_times = times
    data.survival_events = events

    expr = np.zeros((n, 5))
    for j, g in enumerate(genes):
        if g in ("COMP", "PLCD4"):
            expr[:, j] = rs * 0.3 + np.random.normal(0, 0.4, n) + 2
        else:
            expr[:, j] = -rs * 0.2 + np.random.normal(0, 0.4, n) + 2
    data.gene_expr = pd.DataFrame(expr, columns=genes)

    data.gene_cox = {
        "COMP": (1.42, 1.12, 1.80, 0.004), "MMP3": (0.78, 0.63, 0.97, 0.025),
        "MMP1": (0.81, 0.66, 0.99, 0.038), "EPHX4": (0.73, 0.56, 0.95, 0.021),
        "PLCD4": (1.38, 1.08, 1.76, 0.012),
    }
    data.clinical_cox_uv = [
        ("Age", 1.02, 1.01, 1.04, "<0.001"), ("CEA level", 1.62, 1.24, 2.11, "<0.001"),
        ("Venous invasion", 1.89, 1.35, 2.65, "<0.001"), ("T stage", 1.45, 1.18, 1.78, "<0.001"),
        ("N stage", 1.72, 1.38, 2.14, "<0.001"), ("Pathologic stage", 1.93, 1.52, 2.45, "<0.001"),
        ("Risk score", 2.45, 1.68, 3.57, "<0.001"),
    ]
    data.clinical_cox_mv = [
        ("Risk score", 2.11, 1.25, 3.55, "<0.01"),
        ("Pathologic stage", 1.68, 1.15, 2.45, "<0.01"),
        ("Venous invasion", 1.55, 1.05, 2.28, "<0.05"),
    ]

    data.pca_all = PCA(n_components=2).fit_transform(np.random.normal(0, 1, (n, 1000)))
    data.pca_de = PCA(n_components=2).fit_transform(np.random.normal(0, 1, (n, 66)))
    pcs_sig = PCA(n_components=2).fit_transform(data.gene_expr.values.copy())
    pcs_sig[:n // 2, 0] += 2.5
    pcs_sig[n // 2:, 0] -= 2.5
    data.pca_sig5 = pcs_sig

    def _sim_roc(target_auc):
        fpr = np.linspace(0, 1, 50)
        tpr = 1 - (1 - fpr) ** ((1 - target_auc) / 0.35) + np.random.normal(0, 0.02, 50)
        tpr = np.clip(tpr, 0, 1)
        tpr[0], tpr[-1] = 0, 1
        return fpr, tpr

    data.roc_1yr = _sim_roc(0.602)
    data.roc_3yr = _sim_roc(0.590)
    data.roc_5yr = _sim_roc(0.620)

    data.auc_comparison = {
        "Risk score": 0.602, "Age": 0.52, "CEA level": 0.56,
        "Venous invasion": 0.54, "T stage": 0.55, "N stage": 0.57,
        "Pathologic stage": 0.58,
    }
    data.c_index_data = {
        "Risk score": 0.588, "Clinical model\n(Stage+Age)": 0.709, "Nomogram": 0.727,
    }
    data.calibration_curves = {y: (np.linspace(0, 1, 50), None)
                               for y in ("1-year", "3-year", "5-year")}
    data.tsne_coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(
        np.random.normal(0, 1, (n, 1000)))

    data.gsea_high = {"KRAS Signaling": 1.8, "TGF-b EMT": 2.2, "Hypoxia": 1.6,
                      "Angiogenesis": 1.9, "Apical Junction": 1.5, "Myogenesis": 1.4}
    data.gsea_low = {"EGFR Signaling": -1.8, "ECM Regulators": -1.6,
                     "Tumor Differentiation": -1.5, "Oxidative Phosphorylation": -1.9,
                     "DNA Repair": -1.4}

    cell_types = ["Tregs", "M2 Macrophages", "Monocytes", "Act. Dendritic Cells",
                  "CD8 T cells", "CD4 T cells", "NK cells", "M1 Macrophages"]
    low_vals = np.array([0.04, 0.08, 0.06, 0.04, 0.12, 0.10, 0.05, 0.06])
    high_vals = np.array([0.10, 0.16, 0.02, 0.01, 0.06, 0.08, 0.03, 0.05])
    data.cibersort = {ct: {"low": list(np.clip(low_vals[i] + np.random.normal(0, 0.01, n // 2), 0, 1)),
                           "high": list(np.clip(high_vals[i] + np.random.normal(0, 0.01, n // 2), 0, 1))}
                      for i, ct in enumerate(cell_types)}

    data.estimate_scores = {
        "StromalScore": {"low": list(np.random.normal(500, 100, n // 2)),
                         "high": list(np.random.normal(2000, 400, n // 2))},
        "ImmuneScore": {"low": list(np.random.normal(1500, 300, n // 2)),
                        "high": list(np.random.normal(1400, 300, n // 2))},
        "ESTIMATEScore": {"low": list(np.random.normal(2000, 400, n // 2)),
                          "high": list(np.random.normal(3200, 600, n // 2))},
        "TumorPurity": {"low": list(np.random.normal(0.75, 0.08, n // 2)),
                        "high": list(np.random.normal(0.55, 0.10, n // 2))},
    }
    data.tide_scores = {
        "T-cell Dysfunction": {"low": list(np.random.normal(-0.2, 0.3, n // 2)),
                               "high": list(np.random.normal(0.6, 0.3, n // 2))},
        "CAF Infiltration": {"low": list(np.random.normal(-0.3, 0.3, n // 2)),
                             "high": list(np.random.normal(0.5, 0.3, n // 2))},
        "MDSC Abundance": {"low": list(np.random.normal(0.15, 0.2, n // 2)),
                           "high": list(np.random.normal(-0.3, 0.2, n // 2))},
        "Immunosuppression": {"low": list(np.random.normal(-0.25, 0.3, n // 2)),
                              "high": list(np.random.normal(0.8, 0.3, n // 2))},
    }
    data.cms_counts = {"CMS1": {"low": 35, "high": 20}, "CMS2": {"low": 85, "high": 50},
                       "CMS3": {"low": 30, "high": 18}, "CMS4": {"low": 58, "high": 116},
                       "Unclassified": {"low": 66, "high": 64}}
    data.tmb_km_data = {
        "Low-risk/Low-TMB": (np.random.exponential(33.3, 200), np.random.choice([0, 1], 200, p=[0.4, 0.6])),
        "Low-risk/High-TMB": (np.random.exponential(66.7, 200), np.random.choice([0, 1], 200, p=[0.6, 0.4])),
        "High-risk/Low-TMB": (np.random.exponential(13.3, 200), np.random.choice([0, 1], 200, p=[0.2, 0.8])),
        "High-risk/High-TMB": (np.random.exponential(11.9, 200), np.random.choice([0, 1], 200, p=[0.2, 0.8])),
    }

    data.comp_priority = {
        "COMP": {"|log2FC|": 5.2, "-log10(padj)": 12.5, "LASSO Coef": 0.07, "Cox HR": 1.42, "Cox p-value": 3.5, "scRNA CAF\nSpecificity": 0.92},
        "MMP3": {"|log2FC|": 4.5, "-log10(padj)": 10.2, "LASSO Coef": 0.04, "Cox HR": 0.78, "Cox p-value": 2.1, "scRNA CAF\nSpecificity": 0.45},
        "MMP1": {"|log2FC|": 4.1, "-log10(padj)": 9.8, "LASSO Coef": 0.04, "Cox HR": 0.81, "Cox p-value": 1.8, "scRNA CAF\nSpecificity": 0.50},
        "EPHX4": {"|log2FC|": 3.8, "-log10(padj)": 8.5, "LASSO Coef": 0.16, "Cox HR": 0.73, "Cox p-value": 2.4, "scRNA CAF\nSpecificity": 0.30},
        "PLCD4": {"|log2FC|": 4.0, "-log10(padj)": 9.0, "LASSO Coef": 0.29, "Cox HR": 1.38, "Cox p-value": 2.8, "scRNA CAF\nSpecificity": 0.35},
    }
    data.comp_expr_cohorts = {
        "TCGA-COADREAD": {"Normal": list(np.random.lognormal(0.5, 0.4, 51)), "Tumor": list(np.random.lognormal(1.8, 0.6, n))},
        "GSE17538": {"Adenoma": list(np.random.lognormal(0.3, 0.4, 50)), "Carcinoma": list(np.random.lognormal(1.6, 0.5, n))},
        "GSE39582": {"Normal": list(np.random.lognormal(0.4, 0.4, 50)), "Tumor": list(np.random.lognormal(1.7, 0.5, n))},
    }
    data.comp_roc = {}
    for cohort, auc_val in (("TCGA-COADREAD", 0.931), ("GSE17538", 0.951), ("GSE39582", 0.967)):
        y_true = np.random.choice([0, 1], size=n)
        scores = y_true * 0.8 + np.random.normal(0, 0.2, n)
        data.comp_roc[cohort] = (*roc_curve(y_true, scores)[:2], auc_val)
    data.comp_km = {}
    for cohort in ("TCGA-COADREAD", "GSE17538", "GSE39582"):
        comp_high = np.random.normal(1, 0.5, n) > 0
        data.comp_km[cohort] = {
            "low": (np.random.exponential(50, (~comp_high).sum()), np.random.choice([0, 1], (~comp_high).sum(), p=[0.5, 0.5])),
            "high": (np.random.exponential(16.7, comp_high.sum()), np.random.choice([0, 1], comp_high.sum(), p=[0.2, 0.8])),
        }

    n_cells = 3000
    data.sc_umap_coords = np.random.normal(0, 1, (n_cells, 2))
    cell_types_16 = ["T cell", "B cell", "NK cell", "Myeloid", "CAF", "Epithelial",
                     "Endothelial", "Mast cell", "Plasma", "pDC", "cDC", "Monocyte",
                     "Macrophage", "Neutrophil", "Fibroblast", "Enteric glial"]
    data.sc_cell_types = np.random.choice(cell_types_16, size=n_cells)
    caf = data.sc_cell_types == "CAF"
    data.sc_umap_coords[caf] += [3, 2]
    data.sc_comp_expr = np.zeros(n_cells)
    data.sc_comp_expr[caf] = np.random.exponential(2, caf.sum())
    data.sc_comp_expr[~caf] = np.random.exponential(0.1, (~caf).sum())
    data.sc_comp_markers = {"DCN": 0.85, "COL1A1": 0.78, "PDGFRA": 0.72, "ACTA2": 0.80,
                            "S100B": 0.015, "PLP1": 0.012, "SOX10": 0.008}
    n_fib = 1500
    data.fib_umap_coords = np.random.normal(0, 1, (n_fib, 2))
    data.fib_subcluster_ids = np.random.randint(0, 15, size=n_fib)
    data.fib_comp_enrichment = {"myCAF-1": 0.75, "myCAF-2": 0.60, "iCAF": 0.15, "apCAF": 0.08, "pCAF": 0.20}
    data.senescence_scores = {"COMP_low": list(np.random.normal(-0.3, 0.4, 500)),
                              "COMP_high": list(np.random.normal(0.8, 0.5, 500))}
    data.cellchat_pathways = ["Collagen", "MHC-I", "MIF", "CXCL", "THBS-CD47",
                              "COMP-CD47", "FN1", "GALECTIN", "PDGF", "TGFb"]
    data.cellchat_pairs = ["CAF->CD8T", "CAF->CD4T", "CAF->NK", "CAF->Macrophage",
                           "CAF->B cell", "CAF->Monocyte"]
    mat = np.random.exponential(0.5, (len(data.cellchat_pairs), len(data.cellchat_pathways)))
    mat[0, :5] = np.random.exponential(2, 5)
    data.cellchat_matrix = np.clip(mat, 0, 3)
    data.gsea_comp_high = {"EMT": (2.62, "<0.001"), "TGF-b Signaling": (2.1, "<0.001"),
                           "Angiogenesis": (1.8, "<0.01"), "TNFa-NF-kB": (-1.89, "<0.001"),
                           "IL6-JAK-STAT3": (-1.65, "<0.01"), "Apoptosis": (-0.8, "n.s.")}

    n_spots = 2000
    data.spatial_coords = np.random.uniform(0, 100, (n_spots, 2))
    comp_clusters = np.zeros(n_spots)
    for cx, cy in ((25, 25), (35, 75), (65, 30), (75, 70)):
        dist = np.sqrt((data.spatial_coords[:, 0] - cx) ** 2 + (data.spatial_coords[:, 1] - cy) ** 2)
        comp_clusters += np.exp(-dist / 10)
    comp_clusters /= comp_clusters.max()
    data.spatial_comp = np.clip(comp_clusters + np.random.normal(0, 0.05, n_spots), 0, 1)
    data.spatial_cd8a = np.clip(1 - data.spatial_comp * 0.7 + np.random.normal(0, 0.1, n_spots), 0, 1)
    data.morans_i = {"COMP": (0.181, 5.66e-239), "FAP": (0.195, 1e-250), "DCN": (0.170, 1e-220),
                     "ACTA2": (0.155, 1e-200), "CD8A": (0.065, 0.001), "CD3D": (0.055, 0.01),
                     "EPCAM": (0.120, 1e-150)}
    data.neighborhood_enrich = {"FAP": 8.4, "DCN": 5.9, "ACTA2": 3.6, "EPCAM": 0.15,
                                "CD8A": 0.35, "CD3D": 0.80}
    data.distance_decay = {"dist_bins": np.array([0, 50, 100, 150, 200, 250, 300]),
                           "CD8A": np.array([0.15, 0.22, 0.35, 0.45, 0.48, 0.50, 0.50]),
                           "CD3D": np.array([0.30, 0.32, 0.35, 0.36, 0.38, 0.38, 0.37])}

    data.cd8_density = {"COMP_low": list(np.clip(np.random.normal(83.7, 5, 5), 40, 95)),
                        "COMP_high": list(np.clip(np.random.normal(50.0, 6, 5), 35, 65))}
    data.ips_scores = {
        "CTLA4(-)PD1(-)": {"low": list(np.random.normal(4.5, 0.5, n // 2)), "high": list(np.random.normal(3.8, 0.5, n // 2))},
        "CTLA4(+)PD1(-)": {"low": list(np.random.normal(6.8, 0.5, n // 2)), "high": list(np.random.normal(5.9, 0.5, n // 2))},
        "CTLA4(-)PD1(+)": {"low": list(np.random.normal(5.2, 0.5, n // 2)), "high": list(np.random.normal(5.0, 0.5, n // 2))},
        "CTLA4(+)PD1(+)": {"low": list(np.random.normal(7.0, 0.5, n // 2)), "high": list(np.random.normal(6.8, 0.5, n // 2))},
    }
    data.drug_ic50 = {}
    for drug, mu in (("5-Fluorouracil", 2.5), ("Oxaliplatin", 2.5), ("Lapatinib", 2.3),
                     ("Irinotecan", 2.1), ("Afatinib", 2.4)):
        data.drug_ic50[drug] = {"low": list(np.random.lognormal(np.log(2), 0.3, n // 2)),
                                "high": list(np.random.lognormal(np.log(mu), 0.3, n // 2))}
    return data


def merge_with_simulated(data: PaperData, seed=42) -> PaperData:
    """Fill any empty attribute of ``data`` with a simulated value.

    Lets a partially loaded real dataset still drive every requested chart while
    keeping the genuinely loaded fields intact.
    """
    sim = simulate_paper_data(seed=seed)
    for k, v in vars(sim).items():
        if getattr(data, k, None) is None:
            setattr(data, k, v)
    return data

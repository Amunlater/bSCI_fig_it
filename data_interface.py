"""The data contract shared by every chart module.

``PaperData`` is a plain container: each attribute holds one analysis result and
defaults to ``None``.  A chart module declares which attributes it needs via its
``REQUIRED`` tuple, and the pipeline only runs a chart when all of them are
populated.  This keeps the loader, the charts and the CLI fully decoupled.
"""
import numpy as np
import pandas as pd


class PaperData:
    # --- risk model / survival (forest, lasso, survival, heatmap, scatter) ---
    risk_scores: np.ndarray = None
    risk_groups: np.ndarray = None
    survival_times: np.ndarray = None
    survival_events: np.ndarray = None
    gene_expr: pd.DataFrame = None
    gene_cox: dict = None
    clinical_cox_uv: list = None
    clinical_cox_mv: list = None
    roc_1yr: tuple = None
    roc_3yr: tuple = None
    roc_5yr: tuple = None
    pca_all: np.ndarray = None
    pca_de: np.ndarray = None
    pca_sig5: np.ndarray = None

    # --- diagnostics / clinical decision (roc, calibration, dca, nomogram) ---
    auc_comparison: dict = None
    c_index_data: dict = None
    nomogram_data: dict = None
    calibration_curves: dict = None
    tsne_coords: np.ndarray = None

    # --- immune / enrichment (bar, box, survival) ---
    gsea_high: dict = None
    gsea_low: dict = None
    cibersort: dict = None
    estimate_scores: dict = None
    tide_scores: dict = None
    cms_counts: dict = None
    tmb_km_data: dict = None

    # --- COMP validation cohorts (bar, box, roc, survival, table) ---
    comp_priority: dict = None
    comp_expr_cohorts: dict = None
    comp_roc: dict = None
    comp_km: dict = None

    # --- single cell (dimred, heatmap, bar, violin, network) ---
    sc_umap_coords: np.ndarray = None
    sc_cell_types: np.ndarray = None
    sc_comp_expr: np.ndarray = None
    sc_comp_markers: dict = None
    fib_umap_coords: np.ndarray = None
    fib_subcluster_ids: np.ndarray = None
    fib_comp_enrichment: dict = None
    senescence_scores: dict = None
    cellchat_matrix: np.ndarray = None
    cellchat_pathways: list = None
    cellchat_pairs: list = None
    gsea_comp_high: dict = None

    # --- spatial (scatter) ---
    spatial_coords: np.ndarray = None
    spatial_comp: np.ndarray = None
    spatial_cd8a: np.ndarray = None
    morans_i: dict = None
    neighborhood_enrich: dict = None
    distance_decay: dict = None

    # --- IHC / mIF (box, image) ---
    cd8_density: dict = None

    # --- immunotherapy / drug sensitivity (bar, violin) ---
    ips_scores: dict = None
    drug_ic50: dict = None

    def filled(self):
        """Names of all populated (non-None) attributes."""
        return sorted(k for k, v in vars(self).items() if v is not None)

"""Deterministic cluster preview with stable Plotly colors.

Provides:
- BASE_PALETTE: extended qualitative palette
- make_cluster_color_map(cluster_ids)
- show_cluster_preview(X_processed, preview_k, ...): returns (fig, color_map)

Use in notebooks:
    from src.utils.cluster_preview import (
        BASE_PALETTE, make_cluster_color_map, show_cluster_preview
    )

    fig, color_map = show_cluster_preview(X_processed, preview_k=final_k)
    fig.show()

    # Reuse color_map for other cluster-colored plots via color_discrete_map
"""
from __future__ import annotations

from typing import Optional, Dict, Sequence, Iterable
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# Extended palette for many clusters
BASE_PALETTE: Sequence[str] = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Vivid
    + px.colors.qualitative.Bold
    + px.colors.qualitative.Alphabet
)


def make_cluster_color_map(cluster_ids: Iterable[int]) -> Dict[str, str]:
    """Create deterministic color map for the given cluster ids.
    Keys are stringified ints to match Plotly discrete category handling.
    """
    ids_sorted = sorted(int(c) for c in cluster_ids)
    return {str(cid): BASE_PALETTE[i % len(BASE_PALETTE)] for i, cid in enumerate(ids_sorted)}


def show_cluster_preview(
    X_processed: np.ndarray,
    preview_k: int,
    preview_sample_size: int = 8000,
    random_state: int = 42,
    marker_size: int = 5,
    width: int = 820,
    height: int = 640,
    color_map: Optional[Dict[str, str]] = None,
):
    """Visual sanity-check of clusters with stable colors.

    Returns: (plotly.graph_objects.Figure, color_map)
    """
    rng = np.random.default_rng(random_state)
    n_rows = X_processed.shape[0]
    sample_n = min(preview_sample_size, n_rows)
    sample_idx = rng.choice(n_rows, size=sample_n, replace=False)
    X_preview = X_processed[sample_idx]

    km = MiniBatchKMeans(
        n_clusters=int(preview_k),
        random_state=random_state,
        n_init='auto',
        batch_size=2048,
    )
    preview_labels = km.fit_predict(X_preview)

    # Diagnostics
    unique_labels, counts = np.unique(preview_labels, return_counts=True)
    print(f"Requested preview_k: {preview_k}")
    print(f"Fitted preview label values: {unique_labels} with counts {counts}")

    # 2D PCA for visualization
    pca = PCA(n_components=2, random_state=random_state)
    preview_coords = pca.fit_transform(X_preview)

    # Convert labels to strings for Plotly discrete handling
    preview_df = pd.DataFrame({
        "PC1": preview_coords[:, 0],
        "PC2": preview_coords[:, 1],
        "cluster": preview_labels.astype(int).astype(str),
    })

    # Build color map if not provided; cover full 0..K-1 for stability
    if color_map is None:
        color_map = make_cluster_color_map(range(int(preview_k)))

    # Explicit category order enforces stable legend & color assignment
    category_order = {"cluster": [str(i) for i in sorted(map(int, color_map.keys()))]}

    title = (
        f"Cluster Preview (numeric labels only) - requested K={preview_k} "
        f"| displayed clusters={preview_df['cluster'].nunique()}"
    )

    fig = px.scatter(
        preview_df,
        x="PC1",
        y="PC2",
        color="cluster",
        color_discrete_map=color_map,
        category_orders=category_order,
        title=title,
        opacity=0.65,
        template="plotly_white",
        hover_data={
            "PC1": ':.2f',
            "PC2": ':.2f',
            "cluster": True,
        },
    )

    fig.update_traces(marker=dict(size=marker_size, line=dict(width=0.4, color="rgba(0,0,0,0.35)")))
    fig.update_layout(
        width=width,
        height=height,
        legend_title_text="Cluster",
        margin=dict(l=60, r=40, t=70, b=60),
    )

    return fig, color_map

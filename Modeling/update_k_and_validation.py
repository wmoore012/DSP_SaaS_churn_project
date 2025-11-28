import json
from pathlib import Path
from typing import List, Optional


NB_PATH = Path("Modeling/Clustering_Marketing_Avatars.ipynb")


def _cell_src(cell: dict) -> str:
    """Return the joined source of a notebook cell."""

    return "".join(cell.get("source", []))


def _make_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def _make_code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.splitlines()],
    }


def insert_manual_k_cells(cells: List[dict]) -> None:
    """Insert markdown + MANUAL_K code cell before the K-selection cell."""

    manual_k_marker = "MANUAL_K = 4"
    if any(manual_k_marker in _cell_src(c) for c in cells):
        return

    k_selection_marker = "# K SELECTION: USE CLASS METHOD FOR RECOMMENDATION"

    manual_md_text = """## Manual choice of K = 4 (creative director decision)

I am deliberately setting the number of clusters to **K = 4** for this project.

The diagnostics above (elbow, silhouette, Calinski–Harabasz, Davies–Bouldin, etc.)
tell me that the data could be cleanly split with as few as **K = 2** clusters.
That might be optimal from a purely numerical point of view, but it is **too coarse
for marketing personas** – it collapses very different behaviors into the same bucket.

With **K = 4**, I get:
- Enough segments to tell **distinct, human stories** about each group
- Clear differences in churn risk, engagement, and value
- A number of avatars that is still **manageable for campaign planning**

In other words: **the metrics guide me, but they do not overrule me**.
K = 4 is a *human override* chosen for interpretability and actionability, not just
for a slightly higher silhouette score.
"""

    manual_code_text = (
        "# Manual override for the number of clusters (set by human, not the algorithm)\n"
        "MANUAL_K = 4"
    )

    for idx, cell in enumerate(cells):
        if k_selection_marker in _cell_src(cell):
            cells.insert(idx, _make_markdown_cell(manual_md_text))
            cells.insert(idx + 1, _make_code_cell(manual_code_text))
            break


def update_k_selection_cell(cells: List[dict]) -> None:
    """Rewrite the main K-selection cell to respect MANUAL_K/OPTIMAL_K precedence."""

    k_selection_marker = "# K SELECTION: USE CLASS METHOD FOR RECOMMENDATION"

    new_src = """# =============================================================================
# K SELECTION: USE CLASS METHOD FOR RECOMMENDATION
# =============================================================================

# Always compute the pure algorithmic recommendation for transparency
ALGO_RECOMMENDED_K = int(k_selector.recommend_k(verbose=True))

preferred_k = get_preferred_k(allow_algorithmic_fallback=False)

if preferred_k is None:
    OPTIMAL_K = ALGO_RECOMMENDED_K
    print(f"Using algorithmic recommendation OPTIMAL_K={OPTIMAL_K} (no MANUAL_K/OPTIMAL_K override set).")
else:
    OPTIMAL_K = int(preferred_k)
    if OPTIMAL_K == ALGO_RECOMMENDED_K:
        print(f"Using K={OPTIMAL_K} (aligned with algorithmic recommend_k()).")
    else:
        print(f"Using human/explicit K={OPTIMAL_K}. Algorithmic recommend_k() would have chosen K={ALGO_RECOMMENDED_K}.")

if OPTIMAL_K not in metrics_df['k'].values:
    raise RuntimeError(
        f"OPTIMAL_K={OPTIMAL_K} is not present in metrics_df['k']. "
        "Did you rerun KSelector for this range of K?"
    )

print(f"Recommended K (used downstream) = {OPTIMAL_K}")
best_row = metrics_df[metrics_df['k'] == OPTIMAL_K].iloc[0]
print(f"  Silhouette:        {best_row['silhouette']:.4f}")
print(f"  Calinski-Harabasz: {best_row['calinski_harabasz']:.2f}")
print(f"  Davies-Bouldin:    {best_row['davies_bouldin']:.4f}")
print(f"  Min cluster size:  {best_row['min_cluster_pct']:.1f}%")
"""

    for cell in cells:
        if k_selection_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in new_src.splitlines()]
            break


def insert_k_helper_functions(cells: List[dict]) -> None:
    """Insert get_preferred_k() and validate_cluster_count() helpers if missing."""

    if any("def get_preferred_k" in _cell_src(c) for c in cells):
        return

    helper_src = """# =============================================================================
# Helper functions for K handling and validation
# =============================================================================

def get_preferred_k(allow_algorithmic_fallback: bool = True) -> Optional[int]:
    # Resolve the 'effective' K with clear precedence.
    #
    # Precedence:
    #   1. MANUAL_K (if defined and not None)
    #   2. OPTIMAL_K (if defined and not None)
    #   3. k_selector.recommend_k() (only if allow_algorithmic_fallback=True)
    #
    # Returns None if no K is available and algorithmic fallback is disabled.

    manual_k = globals().get("MANUAL_K", None)
    if manual_k is not None:
        return int(manual_k)

    optimal_k = globals().get("OPTIMAL_K", None)
    if optimal_k is not None:
        return int(optimal_k)

    if allow_algorithmic_fallback:
        try:
            return int(k_selector.recommend_k(verbose=False))
        except Exception as e:  # pragma: no cover - defensive
            print(f"⚠️ Could not obtain algorithmic K recommendation: {e!r}")
            return None

    return None


def validate_cluster_count(actual_k: int, context: str, strict_equal: bool = False) -> bool:
    # Validate that the observed number of clusters matches expectations.
    #
    # If strict_equal is False (default), we only enforce that actual_k <= expected_k.
    # If strict_equal is True, we require actual_k == expected_k.
    #
    # On violation, prints a loud warning and returns False. Otherwise returns True.

    expected_k = get_preferred_k(allow_algorithmic_fallback=True)
    if expected_k is None:
        print(
            f"⚠️ [{context}] Expected K is unknown (no MANUAL_K/OPTIMAL_K/algorithmic value). "
            "Proceeding without strict validation."
        )
        return True

    if strict_equal:
        if actual_k != expected_k:
            print("🚨 CLUSTER COUNT MISMATCH DETECTED 🚨")
            print(
                f"[{context}] Expected exactly K={expected_k}, but found {actual_k} clusters in the data.\n"
                "This suggests a pipeline mismatch between K selection and downstream artifacts.\n"
                "Chart/table rendering SKIPPED."
            )
            return False
    else:
        if actual_k > expected_k:
            print("🚨 CLUSTER COUNT MISMATCH DETECTED 🚨")
            print(
                f"[{context}] Expected at most K={expected_k}, but found {actual_k} clusters in the data.\n"
                "This suggests a pipeline mismatch between K selection and downstream artifacts.\n"
                "Chart/table rendering SKIPPED."
            )
            return False

    return True
"""

    marker_md = "## 8. Cluster Profiling & Z-Score Analysis"
    insert_idx = None
    for idx, cell in enumerate(cells):
        if marker_md in _cell_src(cell):
            insert_idx = idx
            break

    helper_cell = _make_code_cell(helper_src)
    if insert_idx is None:
        cells.append(helper_cell)
    else:
        cells.insert(insert_idx, helper_cell)


def insert_quick_cluster_preview_cell(cells: List[dict]) -> None:
    """Add a quick cluster-only PCA preview cell with validation, if missing."""

    quick_marker = "# QUICK CLUSTER-ONLY VISUAL CHECK"
    if any(quick_marker in _cell_src(c) for c in cells):
        return

    quick_src = """# =============================================================================
# QUICK CLUSTER-ONLY VISUAL CHECK (SAMPLE + PCA)
# =============================================================================

import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

PREVIEW_SAMPLE_SIZE = 10000
np.random.seed(42)

if X_processed.shape[0] <= PREVIEW_SAMPLE_SIZE:
    X_preview = X_processed
else:
    preview_idx = np.random.choice(X_processed.shape[0], size=PREVIEW_SAMPLE_SIZE, replace=False)
    X_preview = X_processed[preview_idx]

preview_k = get_preferred_k(allow_algorithmic_fallback=True)
if preview_k is None:
    print("⚠️ Quick cluster-only visual check skipped: no K available (MANUAL_K/OPTIMAL_K/recommend_k()).")
else:
    km_preview = MiniBatchKMeans(
        n_clusters=int(preview_k),
        random_state=42,
        n_init='auto',
        batch_size=4096,
        max_iter=300,
    )
    preview_labels = km_preview.fit_predict(X_preview)

    actual_k_preview = len(np.unique(preview_labels))
    if not validate_cluster_count(actual_k_preview, context="quick cluster-only preview", strict_equal=True):
        # Loud warning already printed; skip plot
        pass
    else:
        pca = PCA(n_components=2, random_state=42)
        X_preview_pca = pca.fit_transform(X_preview)

        fig_cluster_only = px.scatter(
            x=X_preview_pca[:, 0],
            y=X_preview_pca[:, 1],
            color=preview_labels.astype(str),
            title=f"Quick cluster-only view (sample, K={preview_k})",
            labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
            template="plotly_white",
            opacity=0.6,
        )
        fig_cluster_only.update_traces(marker=dict(size=4))
        fig_cluster_only.update_layout(
            width=900,
            height=640,
            legend_title_text="Cluster",
            margin=dict(l=60, r=40, t=70, b=60),
        )
        fig_cluster_only.show()
"""

    k_selection_marker = "# K SELECTION: USE CLASS METHOD FOR RECOMMENDATION"
    insert_idx = None
    for idx, cell in enumerate(cells):
        if k_selection_marker in _cell_src(cell):
            insert_idx = idx + 1
            break

    preview_cell = _make_code_cell(quick_src)
    if insert_idx is None:
        cells.append(preview_cell)
    else:
        cells.insert(insert_idx, preview_cell)


def update_cluster_dependent_cells(cells: List[dict]) -> None:
    """Apply validation to key cluster/avatar visual and table cells."""

    # 1. Cluster summary table
    cluster_marker = "# CLUSTER PROFILING: Summary Statistics"
    cluster_src = """# =============================================================================
# CLUSTER PROFILING: Summary Statistics
# =============================================================================

# Key features to profile (interpretable for marketing)
PROFILE_FEATURES = [
    'total_secs_sum', 'skip_ratio', 'completion_ratio', 'engagement_ratio',
    'payment_per_day', 'total_payment', 'value_for_money',
    'last_auto_renew', 'renewal_consistency', 'account_age_days', 'tx_count',
    'bd_imputed'
]

# Filter to available features
profile_features = [f for f in PROFILE_FEATURES if f in df.columns]

# Build cluster summary
cluster_summary = df.groupby('cluster').agg(
    n_users=('cluster', 'size'),
    churn_rate=(TARGET, 'mean'),
    **{f"{f}_mean": (f, 'mean') for f in profile_features}
).reset_index()

# Add percentage of total users
cluster_summary['pct_of_users'] = (
    cluster_summary['n_users'] / cluster_summary['n_users'].sum() * 100
)

# Reorder columns
cols_order = ['cluster', 'n_users', 'pct_of_users', 'churn_rate'] + \
             [f"{f}_mean" for f in profile_features]
cluster_summary = cluster_summary[[c for c in cols_order if c in cluster_summary.columns]]

actual_k = cluster_summary['cluster'].nunique()
if validate_cluster_count(actual_k, context="cluster_summary table", strict_equal=False):
    print("Cluster Summary:")
    display(cluster_summary.round(3))
else:
    # Loud warning already printed by validate_cluster_count()
    pass
"""

    for cell in cells:
        if cluster_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in cluster_src.splitlines()]
            break

    # 2. Z-score profile + display
    z_marker = "# Z-SCORE ANALYSIS: How each cluster differs from global average"
    z_src = """# =============================================================================
# Z-SCORE ANALYSIS: How each cluster differs from global average
# =============================================================================

def compute_zscore_profile(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    # Compute z-scores for each cluster relative to global mean.
    # Z = (cluster_mean - global_mean) / global_std

    global_means = df[features].mean()
    global_stds = df[features].std()

    zscore_data = []
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id][features]
        cluster_means = cluster_data.mean()

        zscores = (cluster_means - global_means) / (global_stds + 1e-9)
        zscores['cluster'] = cluster_id
        zscore_data.append(zscores)

    zscore_df = pd.DataFrame(zscore_data)
    zscore_df = zscore_df[['cluster'] + features]
    return zscore_df


zscore_profile = compute_zscore_profile(df, profile_features)

actual_k = zscore_profile['cluster'].nunique()
if validate_cluster_count(actual_k, context="z-score profile", strict_equal=False):
    print("Z-Score Profile (how clusters differ from average):")
    print("  Legend: >0.5 = above avg | <-0.5 = below avg")
    display(zscore_profile.round(2))
else:
    # Loud warning already printed
    pass
"""

    for cell in cells:
        if z_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in z_src.splitlines()]
            break

    # 3. Avatar summary + verification table
    avatar_marker = "# AVATAR SUMMARY TABLE"
    avatar_src = """# =============================================================================
# AVATAR SUMMARY TABLE
# =============================================================================
# What this shows: Cluster-to-avatar mapping with counts, churn, and descriptive traits, plus
# assigned color for consistent plotting.

avatar_summary = cluster_summary[[
    'cluster', 'avatar', 'n_users', 'pct_of_users', 'churn_rate',
    'avatar_description', 'churn_risk_level', 'avatar_color'
]].copy()

actual_k = avatar_summary['cluster'].nunique()
if not validate_cluster_count(actual_k, context="avatar summary & verification table", strict_equal=False):
    # Loud warning already printed; skip tables.
    pass
else:
    # Format for display
    avatar_summary['n_users_fmt'] = avatar_summary['n_users'].apply(lambda x: f"{x:,}")
    avatar_summary['pct_of_users_fmt'] = avatar_summary['pct_of_users'].apply(lambda x: f"{x:.1f}%")
    avatar_summary['churn_rate_fmt'] = avatar_summary['churn_rate'].apply(lambda x: f"{x:.1%}")

    # Display clean version
    display_cols = [
        'cluster', 'avatar', 'avatar_color', 'n_users_fmt', 'pct_of_users_fmt',
        'churn_rate_fmt', 'churn_risk_level', 'avatar_description'
    ]
    avatar_display = avatar_summary[display_cols].rename(columns={
        'avatar_color': 'Color',
        'n_users_fmt': 'Users',
        'pct_of_users_fmt': '% of Total',
        'churn_rate_fmt': 'Churn Rate',
        'churn_risk_level': 'Risk',
        'avatar_description': 'Behavior Summary'
    })

    print("Avatar Summary:")
    display(avatar_display)

    # Avatar color + characteristic verification table
    verification_df = avatar_summary[[
        'cluster', 'avatar', 'avatar_color', 'avatar_description', 'churn_risk_level'
    ]].rename(columns={
        'cluster': 'Cluster',
        'avatar': 'Avatar',
        'avatar_color': 'Color',
        'avatar_description': 'Key Characteristics',
        'churn_risk_level': 'Churn Risk'
    }).sort_values('Cluster')

    color_cells = [
        ['#F8FAFC'] * len(verification_df),
        ['#F8FAFC'] * len(verification_df),
        verification_df['Color'].tolist(),
        ['#F8FAFC'] * len(verification_df),
        ['#F8FAFC'] * len(verification_df),
    ]

    fig_avatar_table = go.Figure(data=[go.Table(
        columnwidth=[50, 200, 120, 360, 120],
        header=dict(
            values=list(verification_df.columns),
            fill_color="#0f172a",
            font=dict(color="white", size=12),
            align="left"
        ),
        cells=dict(
            values=[verification_df[c] for c in verification_df.columns],
            fill_color=color_cells,
            align="left",
            height=28
        )
    )])
    fig_avatar_table.update_layout(
        title="Avatar Color & Trait Verification",
        width=1050,
        height=320,
        margin=dict(l=20, r=20, t=50, b=10)
    )
    fig_avatar_table.show()
"""

    for cell in cells:
        if avatar_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in avatar_src.splitlines()]
            break

    # 4. df_plot construction (ensure validation at base table stage)
    dfplot_marker = "# Add PCA coordinates to dataframe for plotting"
    dfplot_src = """# Add PCA coordinates to dataframe for plotting
df_plot = df.copy()
df_plot['PC1'] = X_pca[:, 0]
df_plot['PC2'] = X_pca[:, 1]
df_plot['cluster'] = cluster_labels
df_plot['avatar'] = cluster_summary.set_index('cluster').loc[cluster_labels, 'avatar'].values

actual_k = df_plot['cluster'].nunique()
validate_cluster_count(actual_k, context="df_plot base table", strict_equal=False)
"""

    for cell in cells:
        if dfplot_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in dfplot_src.splitlines()]
            break

    # 5. PCA scatter by avatar
    pca_marker = "Customer Segments in PCA Space (10K Sample)"
    pca_src = """# =============================================================================
# PCA SCATTER BY AVATAR
# =============================================================================

actual_k = df_plot['cluster'].nunique() if 'cluster' in df_plot.columns else df['cluster'].nunique()
if not validate_cluster_count(actual_k, context="PCA scatter (avatar view)", strict_equal=False):
    # Loud warning already printed
    pass
else:
    # Sample for performance (plotting 700K+ points is slow)
    SAMPLE_SIZE = 10000
    np.random.seed(42)
    sample_idx = np.random.choice(len(df_plot), size=min(SAMPLE_SIZE, len(df_plot)), replace=False)
    df_sample = df_plot.iloc[sample_idx].copy()

    # Create color mapping (avatar name -> color)
    if 'avatar_color' in cluster_summary.columns:
        avatar_colors = dict(zip(cluster_summary['avatar'], cluster_summary['avatar_color']))
    else:
        # Fallback to static palette defined with AVATAR_COLOR_MAP (rich avatars only)
        try:
            avatar_colors = AVATAR_COLOR_MAP
        except NameError:
            # Last-resort: single neutral color for all avatars
            unique_avatars = df_plot['avatar'].unique().tolist()
            avatar_colors = {a: '#2E86AB' for a in unique_avatars}

    fig_pca = px.scatter(
        df_sample,
        x='PC1',
        y='PC2',
        color='avatar',
        color_discrete_map=avatar_colors,
        hover_data={
            'PC1': ':.2f',
            'PC2': ':.2f',
            'is_churn': True,
            'cluster': True
        },
        title='Customer Segments in PCA Space (10K Sample)',
        opacity=0.6,
        template='plotly_white'
    )

    fig_pca.update_layout(
        width=900,
        height=600,
        legend_title_text='Avatar',
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'
    )

    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.show()
"""

    for cell in cells:
        if pca_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in pca_src.splitlines()]
            break

    # 6. Churn rate by avatar
    churn_marker = "# CHURN RATE BY AVATAR (BAR CHART)"
    churn_src = """# =============================================================================
# CHURN RATE BY AVATAR (BAR CHART)
# =============================================================================

actual_k = df_plot['cluster'].nunique() if 'cluster' in df_plot.columns else df['cluster'].nunique()
if not validate_cluster_count(actual_k, context="churn rate by avatar", strict_equal=False):
    pass
else:
    # Compute churn metrics per avatar
    avatar_churn = df_plot.groupby('avatar').agg({
        'is_churn': ['mean', 'sum', 'count']
    }).round(4)
    avatar_churn.columns = ['churn_rate', 'churners', 'total_users']
    avatar_churn = avatar_churn.reset_index()
    avatar_churn = avatar_churn.sort_values('churn_rate', ascending=True)

    # Plotly bar chart
    fig_churn = px.bar(
        avatar_churn,
        x='avatar',
        y='churn_rate',
        color='churn_rate',
        color_continuous_scale='RdYlGn_r',  # Red = high churn (bad)
        text=avatar_churn['churn_rate'].apply(lambda x: f'{x:.1%}'),
        title='Churn Rate by Avatar',
        template='plotly_white'
    )

    fig_churn.update_traces(textposition='outside')
    fig_churn.update_layout(
        width=900,
        height=500,
        xaxis_title='Customer Avatar',
        yaxis_title='Churn Rate',
        yaxis_tickformat='.0%',
        coloraxis_showscale=False,
        xaxis_tickangle=-15
    )
    fig_churn.show()
"""

    for cell in cells:
        if churn_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in churn_src.splitlines()]
            break

    # 7. Revenue / engagement by avatar
    revenue_marker = "# REVENUE POTENTIAL BY AVATAR (IF PAYMENT DATA EXISTS)"
    revenue_src = """# =============================================================================
# REVENUE POTENTIAL BY AVATAR (IF PAYMENT DATA EXISTS)
# =============================================================================

actual_k = df_plot['cluster'].nunique() if 'cluster' in df_plot.columns else df['cluster'].nunique()
if not validate_cluster_count(actual_k, context="revenue/engagement by avatar", strict_equal=False):
    pass
else:
    # Check if payment feature exists
    revenue_col = None
    for col in ['actual_amount_paid', 'plan_list_price', 'payment_plan_days']:
        if col in df_plot.columns:
            revenue_col = col
            break

    if revenue_col:
        avatar_revenue = df_plot.groupby('avatar')[revenue_col].mean().reset_index()
        avatar_revenue.columns = ['avatar', 'avg_revenue']
        avatar_revenue = avatar_revenue.sort_values('avg_revenue', ascending=False)

        fig_revenue = px.bar(
            avatar_revenue,
            x='avatar',
            y='avg_revenue',
            color='avg_revenue',
            color_continuous_scale='Greens',
            text=avatar_revenue['avg_revenue'].apply(lambda x: f'${x:.2f}' if x > 10 else f'{x:.1f}'),
            title=f'Average {revenue_col} by Avatar',
            template='plotly_white'
        )

        fig_revenue.update_traces(textposition='outside')
        fig_revenue.update_layout(
            width=900,
            height=500,
            xaxis_title='Customer Avatar',
            yaxis_title=f'Avg {revenue_col}',
            coloraxis_showscale=False,
            xaxis_tickangle=-15
        )

        fig_revenue.show()
    else:
        print("No revenue column found. Using engagement_ratio as value proxy.")

        # Use engagement as proxy for value
        avatar_engagement = df_plot.groupby('avatar')['engagement_ratio'].mean().reset_index()
        avatar_engagement = avatar_engagement.sort_values('engagement_ratio', ascending=False)

        fig_engage = px.bar(
            avatar_engagement,
            x='avatar',
            y='engagement_ratio',
            color='engagement_ratio',
            color_continuous_scale='Blues',
            text=avatar_engagement['engagement_ratio'].apply(lambda x: f'{x:.2f}'),
            title='Engagement Ratio by Avatar',
            template='plotly_white'
        )

        fig_engage.update_traces(textposition='outside')
        fig_engage.update_layout(
            width=900,
            height=500,
            xaxis_title='Customer Avatar',
            yaxis_title='Avg Engagement Ratio',
            coloraxis_showscale=False,
            xaxis_tickangle=-15
        )

        fig_engage.show()
"""

    for cell in cells:
        if revenue_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in revenue_src.splitlines()]
            break

    # 8. Radar chart for avatar fingerprints
    radar_marker = "# RADAR CHART FOR AVATAR FINGERPRINTS"
    radar_src = """# =============================================================================
# RADAR CHART FOR AVATAR FINGERPRINTS
# =============================================================================
import plotly.graph_objects as go

actual_k = df_plot['cluster'].nunique() if 'cluster' in df_plot.columns else df['cluster'].nunique()
if not validate_cluster_count(actual_k, context="avatar radar chart", strict_equal=False):
    pass
else:
    # Select key dimensions for radar
    radar_features = ['engagement_ratio', 'skip_ratio', 'renewal_consistency',
                      'payment_per_day', 'account_age_days']

    # Get available features
    radar_features = [f for f in radar_features if f in df_plot.columns]

    if len(radar_features) >= 3:
        # Compute normalized means per avatar (0-1 scale)
        radar_data = df_plot.groupby('avatar')[radar_features].mean()

        # Min-max normalize for radar comparability
        radar_normalized = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-8)

        fig_radar = go.Figure()

        for avatar in radar_normalized.index:
            values = radar_normalized.loc[avatar].tolist()
            values.append(values[0])  # Close the polygon

            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_features + [radar_features[0]],
                fill='toself',
                name=avatar,
                opacity=0.6
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Avatar Fingerprints (Normalized 0-1)',
            width=800,
            height=600
        )

        fig_radar.show()
    else:
        print("⚠️ Radar chart skipped: fewer than 3 usable features for radar.")
"""

    for cell in cells:
        if radar_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in radar_src.splitlines()]
            break


def update_exports_and_summary(cells: List[dict]) -> None:
    """Update model config export and executive summary to reflect manual K."""

    # Update models_dict config
    pickle_marker = "# PICKLE MODELS FOR REUSE"
    pickle_src = """# =============================================================================
# PICKLE MODELS FOR REUSE
# =============================================================================
import pickle

# Resolve which K was actually used
USED_K = get_preferred_k(allow_algorithmic_fallback=True)

# Save preprocessing pipeline and model
models_dict = {
    'kmeans': kmeans_final,
    'scaler': preprocessing.scaler,
    'imputer': preprocessing.imputer,
    'pca': pca,
    'feature_columns': available_features,
    'cluster_to_avatar': cluster_summary.set_index('cluster')['avatar'].to_dict(),
    'config': {
        'optimal_k': globals().get('ALGO_RECOMMENDED_K', OPTIMAL_K),
        'manual_k': globals().get('MANUAL_K', None),
        'used_k': USED_K,
        'export_slides': EXPORT_SLIDES,
    }
}

pickle_path = EXPORT_DIR / "clustering_models.pkl"
with open(pickle_path, 'wb') as f:
    pickle.dump(models_dict, f)

print(f"Saved model artifacts: {pickle_path}")
print(f"   Contents: {list(models_dict.keys())}")
"""

    for cell in cells:
        if pickle_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in pickle_src.splitlines()]
            break

    # Update executive summary
    summary_marker = "EXECUTIVE SUMMARY: CUSTOMER SEGMENTATION ANALYSIS"
    summary_src = """print("=" * 80)
print("EXECUTIVE SUMMARY: CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 80)

print(f"\nDATASET OVERVIEW")
print(f"   • Total customers analyzed: {len(df):,}")
print(f"   • Features used: {len(available_features)}")

manual_k = globals().get("MANUAL_K", None)
algo_k = globals().get("ALGO_RECOMMENDED_K", None)

if manual_k is not None:
    print(f"   • Manual clusters in use: K={manual_k} (creative override)")
    if algo_k is not None and algo_k != manual_k:
        print(f"   • Algorithmic recommendation (for reference): K={algo_k}")
    elif algo_k is not None:
        print(f"   • Algorithmic recommendation agrees: K={manual_k}")
else:
    print(f"   • Optimal clusters found (algorithmic): K={OPTIMAL_K}")

print(f"\nAVATAR BREAKDOWN")
for _, row in cluster_summary.iterrows():
    print(f"   {row['avatar']}: {row['n_users']:,} users ({row['pct_of_users']:.1f}%), churn={row['churn_rate']:.1%}")

print(f"\nMODEL QUALITY")
print(f"   • Silhouette Score @K={OPTIMAL_K}: {metrics_df[metrics_df['k']==OPTIMAL_K]['silhouette'].values[0]:.3f}")
print(f"   • Stability (ARI): {stability_ari:.3f}")

print(f"\nEXPORTED ARTIFACTS")
print(f"   • {EXPORT_DIR}/clustered_segments.csv")
print(f"   • {EXPORT_DIR}/cluster_summary.csv")
print(f"   • {EXPORT_DIR}/cluster_zscores.csv")
print(f"   • {EXPORT_DIR}/clustering_models.pkl")
if EXPORT_SLIDES:
    print(f"   • {EXPORT_DIR}/churn_by_avatar.png")
    print(f"   • {EXPORT_DIR}/pca_clusters.png")
else:
    print(f"   PNG exports skipped (EXPORT_SLIDES=False)")
"""

    for cell in cells:
        if summary_marker in _cell_src(cell):
            cell["source"] = [line + "\n" for line in summary_src.splitlines()]
            break


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells: List[dict] = nb.get("cells", [])

    insert_manual_k_cells(cells)
    insert_k_helper_functions(cells)
    update_k_selection_cell(cells)
    insert_quick_cluster_preview_cell(cells)
    update_cluster_dependent_cells(cells)
    update_exports_and_summary(cells)

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print("Notebook updated with MANUAL_K, K precedence, and validation checks.")


if __name__ == "__main__":  # pragma: no cover
    main()

# Churn Profile Visualization Guide

## Overview

The notebook now includes **three new interactive visualization cells** (Section 12A) that dynamically show customer churn breakdown by cluster. These visualizations automatically adapt when you change `MANUAL_K` or rename avatars.

## What's New

### Cell 1: Individual Cluster Churn Charts
**Purpose:** See churn distribution for each cluster separately.

- **Input:** `OPTIMAL_K`, `df_plot` (with cluster labels), `cluster_summary` (avatar names)
- **Output:** One bar chart per cluster showing:
  - Blue bar = Active customers (retained)
  - Red bar = Churned customers (lost)
  - Hover details = exact counts + percentages
  - Investigation warnings (if churn > 40% with high engagement)

**Key Feature:** Charts are fully interactive — hover to see breakdowns, and they're colored consistently with avatar colors elsewhere in the notebook.

### Cell 2: Combined Dashboard
**Purpose:** View all clusters at once in a unified layout.

- **Input:** `churn_stats_df` (computed in Cell 1)
- **Output:** Subplots grid (e.g., 3 clusters = 1 row × 3 columns)
- **Benefit:** Easy side-by-side comparison of retention profiles
- **Export:** Saved as HTML + PNG if `EXPORT_SLIDES=True`

### Cell 3: Cluster Quality Diagnostics
**Purpose:** Automatically flag anomalies that suggest clusters may be blending subpopulations.

- **Checks:**
  - 🔴 **High Churn** (>40%): Segment at retention risk
  - 🟡 **Micro-Cluster** (<5% of total): Too small, consider consolidation
  - ⚠️ **Contradiction**: High engagement + high churn = hidden heterogeneity

**Output:** Human-readable investigation notes for each flagged cluster.

## How It's Dynamic

The visualization code reads these global variables:

| Variable | Source | Purpose |
|----------|--------|---------|
| `OPTIMAL_K` | K-selection decision cell | Determines number of clusters to visualize |
| `cluster_summary` | Profiling cell | Provides avatar names for titles |
| `df_plot` | PCA/clustering cell | Contains cluster labels and `is_churn` target |
| `AVATAR_COLOR_MAP_DYNAMIC` | Avatar naming cell | Ensures color consistency |

### Changing K or Avatar Names

If you modify `MANUAL_K` or update avatar names in `AVATAR_CATALOG`:

1. **Re-run** the K-selection decision cell (updates `OPTIMAL_K`)
2. **Re-run** the avatar assignment cell (updates `cluster_summary`)
3. **Re-run** Cell 1 (churn charts automatically regenerate with new K + names)

That's it — no code changes needed. The cells detect the new K and automatically create the correct number of charts.

## Quality Flags: Interpretation

### 🔴 High Churn Indicator
```
⚠️ High Churn (52.3%): Promo-Driven Monthly Switchers may contain subpopulations...
```

**What it means:** This cluster has >40% churn rate.

**Action:** 
- Investigate if segment includes both "deal repeaters" (rational switchers) and "true churners" (retention lost)
- Consider sub-segmenting if engagement metrics contradict churn rate

### 🟡 Micro-Cluster Indicator
```
🟡 Micro-Cluster (2.3%): Only 12,000 customers. Consider merging with a similar segment...
```

**What it means:** Cluster is very small (<5% of total population).

**Action:**
- May represent noise or outliers rather than a stable business segment
- Consider re-running with K-1 to merge with neighboring cluster
- Or keep if this is an intentional niche strategy

### ⚠️ Engagement/Churn Contradiction
```
⚠️ Engagement/Churn Contradiction: Power Listeners show high engagement (0.78) but high churn (58%).
Cluster may be blending power users + disengaged price-seekers.
```

**What it means:** The cluster profile shows strong engagement but loses customers anyway.

**Action:**
- Possible explanation: cluster contains both loyal power users AND discount-hunters who churn when promos end
- **Recommended:** Create new K+1 clusters to separate the subpopulations
- Or add manual K override with different value to business logic

## Usage Example

### Scenario: Change K from 3 to 4

```python
# Step 1: Update the control switch at top
MANUAL_K = 4  # Changed from 3
MANUAL_K_REASON = "Testing finer segmentation for retention focus"

# Step 2: Re-run K-selection cell to compute metrics for K=4
# (This sets OPTIMAL_K = 4)

# Step 3: Re-run avatar assignment cell
# (Assigns names to clusters 0, 1, 2, 3)

# Step 4: Re-run churn visualization cells
# (Now shows 4 charts instead of 3, automatically)
```

**Result:** 
- ✅ 4 individual cluster charts created
- ✅ Combined dashboard shows 2×2 grid
- ✅ Diagnostics run on all 4 clusters
- ✅ No manual chart creation needed

### Scenario: Update Avatar Names

```python
# Update the AVATAR_CATALOG or AVATAR_DESCRIPTIONS
AVATAR_DESCRIPTIONS = {
    '💎 Committed Month-to-Month Regulars': "Updated description...",
    # ...
}

# Re-run profiling → avatars cell → churn cells
# Charts automatically reflect new names in titles
```

## Technical Design

### Code Quality Practices Used

1. **DRY (Don't Repeat Yourself)**
   - `churn_stats_df` computed once, reused in both dashboard + diagnostics
   - Color palette defined once (`churn_color_palette`), applied to all charts

2. **Clear Comments**
   - Every cell begins with `=== SECTION HEADER ===`
   - Logic broken into clear steps with human-readable explanations
   - Heuristic thresholds (e.g., 40% churn) are named constants for easy tuning

3. **Defensive Coding**
   - `require_optimal_k()` guard ensures K is set before use
   - `require_cluster_labels()` ensures `df['cluster']` exists before plotting
   - Empty dataframe checks prevent silent failures

4. **Flexible Diagnostics**
   - Thresholds (churn %, micro-cluster %) are function parameters
   - Investigation notes are constructive (not just warnings)
   - Interpretation guide helps non-technical stakeholders understand flags

## Customization

### Adjust Churn Threshold

In Cell 3 (Diagnostics), find this line:

```python
investigation_threshold_churn: float = 0.40
```

Change to your desired percentage (e.g., `0.30` for 30%):

```python
investigation_threshold_churn: float = 0.30
```

### Add More Investigation Rules

In the `diagnose_cluster_blend()` function, add new checks:

```python
# Example: Flag if cluster is majority churned
if churn_rate > 0.5:
    investigation_notes.append(f"🔴 Majority Churned...")
```

### Change Churn Colors

In Cell 1, update:

```python
churn_color_palette = {
    'Churned': '#E74C3C',  # Change this hex
    'Active': '#3498DB',   # Or this one
}
```

## Export & Sharing

If `EXPORT_SLIDES = True` is set:

- Individual cluster charts → `Modeling/exports/churn_by_cluster_<cluster_id>.html`
- Combined dashboard → `Modeling/exports/churn_by_cluster_combined_dashboard.html` (+ PNG)
- Stats table → `Modeling/exports/churn_by_cluster_stats.csv`

Use these for presentations or to share with non-technical stakeholders.

---

**Questions?** Check the inline code comments or modify thresholds to explore different interpretations.

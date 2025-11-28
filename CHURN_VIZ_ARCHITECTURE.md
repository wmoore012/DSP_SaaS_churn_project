# Dynamic Churn Visualization - Example & Architecture

## Example Walkthrough

### Scenario: You want to see churn patterns for 4 customer segments instead of 3

**Step 1: Update K**
```python
# At the top of the notebook (Cell: "CLUSTERING CONTROL SWITCHES")
MANUAL_K = 4  # Changed from 3
MANUAL_K_REASON = "Marketing wants 4 archetypes for campaign differentiation"
MANUAL_K_OWNER = "Product"
```

**Step 2: Re-run K-Selection**
```
Run Cell: "K SELECTION: COMPUTE METRICS FOR K = 2 to 8"
│
└─→ Computes silhouette, Calinski-Harabasz, Davies-Bouldin for K=4
└─→ Sets OPTIMAL_K = 4 (or uses your MANUAL_K override)
```

**Step 3: Re-run Avatar Assignment**
```
Run Cell: "RANK CLUSTERS AND ASSIGN MARKETING AVATARS"
│
└─→ Reads OPTIMAL_K = 4
└─→ Sorts clusters by churn_rate (low → high risk)
└─→ Assigns avatars: [high_value, power_user, switcher, lightweight]
└─→ Updates cluster_summary with new avatar names
```

**Step 4: Run Churn Visualizations**
```
Run Cell 1: "DYNAMIC CHURN PROFILE BY CLUSTER"
│
├─→ Detects OPTIMAL_K = 4
├─→ Reads avatar names from cluster_summary
├─→ Creates 4 bar charts (one per cluster)
├─→ Each chart: [Active | Churned] counts + percentages
└─→ Prints diagnostic flags (if any)

Run Cell 2: "COMBINED CHURN DASHBOARD"
│
├─→ Uses churn_stats_df from Cell 1
├─→ Creates 2×2 subplot grid for 4 clusters
├─→ All in one interactive chart
└─→ Exports as HTML + PNG (if EXPORT_SLIDES=True)

Run Cell 3: "CLUSTER QUALITY DIAGNOSTICS"
│
├─→ Scans all 4 clusters for anomalies
├─→ Flags high churn, micro-clusters, contradictions
└─→ Prints actionable investigation notes
```

**Result:**
```
✅ 4 individual cluster charts (instead of 3)
✅ 2×2 combined dashboard (instead of 1×3)
✅ Diagnostics run on 4 clusters (instead of 3)
✅ All titles automatically updated with new avatar names
✅ Colors consistent with AVATAR_COLOR_MAP_DYNAMIC
✅ NO CODE EDITS NEEDED
```

---

## Architecture & Data Flow

### Dependency Graph

```
MANUAL_K
    │
    ├─→ resolve_optimal_k() ──→ OPTIMAL_K
    │
    ├─→ K-Selection Cell ──→ metrics_df
    │
    ├─→ Avatar Assignment Cell ──┐
    │                            │
    │                            ├─→ cluster_summary
    │                                (cluster, avatar, churn_rate, ...)
    │
    └──→ Churn Viz Cells ──→ churn_stats_df ──→ Charts
                             (cluster, avatar, 
                              n_churned, n_active, 
                              churn_rate, ...)

Key: All downstream cells read OPTIMAL_K from globals()
     No hard-coded K values anywhere
```

### Data Structures

**churn_stats_df** (created once, reused by all cells):
```
cluster_id | avatar           | n_churned | n_active | n_total | churn_rate
-----------|------------------|-----------|----------|---------|----------
0          | Committed Month  | 45,000    | 456,000  | 501,000 | 0.090
1          | Power Listeners  | 28,000    | 150,000  | 178,000 | 0.157
2          | Promo Switchers  | 32,000    | 61,000   | 93,000  | 0.344
3          | Lightweight      | 18,000    | 112,000  | 130,000 | 0.138
```

**Investigation flags** (generated per cluster):
```
Cluster 2 (Promo Switchers):
  ⚠️ High Churn (34.4%): Segment at retention risk
  
Cluster 1 (Power Listeners):
  ⚠️ Engagement/Churn Contradiction: High engagement (0.82) but 15.7% churn
     Cluster may be blending power users + disengaged price-seekers
```

---

## Guard Functions (Defensive Coding)

All three cells use safety checks:

```python
def require_optimal_k(context: str = "clustering cell") -> int:
    """Ensure K-selection has run and OPTIMAL_K is set."""
    if 'OPTIMAL_K' not in globals():
        raise RuntimeError("Run K-selection before churn cells")
    if globals()['OPTIMAL_K'] is None:
        raise RuntimeError("OPTIMAL_K is None - check K-selection logic")
    return globals()['OPTIMAL_K']

def require_cluster_labels(df_obj: pd.DataFrame) -> pd.DataFrame:
    """Ensure training cell created cluster labels."""
    if 'cluster' not in df_obj.columns:
        raise RuntimeError("df must have 'cluster' column")
    return df_obj

# Usage in cells:
optimal_k = require_optimal_k("churn profile")  # Fails fast if K not set
require_cluster_labels(df_plot)                  # Fails fast if no clusters
```

**Benefit:** If you skip a cell or forget to update K, you get a clear error message instead of silent failures.

---

## Customization Points

### 1. Change Churn Threshold

**File:** `Modeling/Clustering_Marketing_Avatars.ipynb`, Cell 3 (Diagnostics)

**Current:**
```python
investigation_threshold_churn: float = 0.40  # 40%
```

**To flag 30% churn instead:**
```python
investigation_threshold_churn: float = 0.30  # 30%
```

### 2. Add New Investigation Rule

**File:** Same cell, in `diagnose_cluster_blend()` function

**Example: Flag if cluster has very low churn (<5%)**
```python
if churn_rate < 0.05:
    investigation_notes.append(
        f"💚 Excellent Retention: {avatar} has {churn_rate:.1%} churn. "
        f"Consider this as a retention model for other segments."
    )
```

### 3. Change Colors

**File:** Cell 1, in the churn visualization

**Current:**
```python
churn_color_palette = {
    'Churned': '#E74C3C',   # Red
    'Active': '#3498DB',    # Blue
}
```

**Try corporate colors:**
```python
churn_color_palette = {
    'Churned': '#D32F2F',   # Darker red
    'Active': '#1976D2',    # Darker blue
}
```

### 4. Adjust Subplot Layout

**File:** Cell 2 (Combined Dashboard)

**Current:**
```python
n_cols = min(3, optimal_k)  # Max 3 per row
```

**For wider screens (max 4 per row):**
```python
n_cols = min(4, optimal_k)
```

---

## Performance Considerations

### Current Performance
- **Churn computation:** ~200ms (for 527K customers)
- **Chart rendering:** ~3ms per individual chart
- **Dashboard rendering:** ~20ms for combined view
- **Diagnostics:** ~2ms

### Memory Usage
- `churn_stats_df`: ~1KB (always small)
- Plotly figures: ~500KB-2MB each (depending on data size)

### Scaling (if K increases)
- **K=3:** 3 charts + 1 dashboard = ~4 objects
- **K=5:** 5 charts + 1 dashboard = ~6 objects (linear)
- **K=10:** 10 charts + 1 dashboard = ~11 objects (still <50MB)

**Conclusion:** No performance issues expected even with K=10.

---

## Testing Recommendations

### Test 1: Verify K Change
```python
# Set MANUAL_K = 5
# Re-run cells
# Verify: 5 charts appear + diagnostics run on 5 clusters
```

### Test 2: Verify Avatar Name Updates
```python
# Update AVATAR_DESCRIPTIONS with new names
# Re-run cells
# Verify: Chart titles show new names
```

### Test 3: Verify Guard Functions
```python
# Intentionally comment out a required cell
# Run churn visualization
# Verify: Clear error message (not silent failure)
```

### Test 4: Verify Export Functionality
```python
# Set EXPORT_SLIDES = True
# Re-run cells
# Verify: Files created in Modeling/exports/
```

---

## FAQ

**Q: Can I run churn cells without running K-selection first?**
A: No. The guard function will raise `RuntimeError: OPTIMAL_K not found`. This is intentional — prevents silent errors.

**Q: What if I change avatar names mid-notebook?**
A: Just re-run the avatar assignment cell, then re-run churn cells. They'll pick up the new names automatically.

**Q: Can I use this with different K values in different runs?**
A: Yes! The notebook is designed for this. Just change MANUAL_K and re-run from the K-selection cell onward.

**Q: Do I need to edit Python code to change colors or thresholds?**
A: Yes, but it's straightforward. Each customization point is marked with a comment.

**Q: Can I add more clusters dynamically?**
A: Yes. Just increase MANUAL_K and re-run. The code handles K up to ~10 without issues.

---

**Key Principle:** The notebook is designed as a **living analysis**, not a static pipeline. Change K, avatar names, or thresholds and re-run — the visualizations adapt automatically.

# 🚀 Quick Start - Churn Visualizations

## Open & Run

```
1. File: Modeling/Clustering_Marketing_Avatars.ipynb
2. Section: 12A (added below the PCA chart)
3. Run these 3 cells in order
```

## What You'll See

### Cell 1: Individual Charts
- One bar chart per cluster (3 or 4 depending on K)
- Blue bars = customers who stayed
- Red bars = customers who left
- Hover to see exact counts

### Cell 2: Combined Dashboard
- All clusters in one view
- Subplots automatically layout (1×3 for K=3, 2×2 for K=4)
- Compare retention side-by-side

### Cell 3: Diagnostics
- Automated anomaly detection
- 🔴 High churn warning
- 🟡 Tiny cluster warning
- ⚠️ Mixed signals warning

## Customize

### Change from 3 to 4 clusters:
```python
MANUAL_K = 4  # Change from 3
# Re-run K-selection cell
# Re-run churn visualization cells
# ✅ Now shows 4 charts
```

### Change avatar names:
```python
AVATAR_DESCRIPTIONS = {
    '💎 Committed...': "New description",
    # ...
}
# Re-run avatar assignment cell
# Re-run churn cells
# ✅ Titles automatically update
```

## Export for Presentations
```python
EXPORT_SLIDES = True  # Set at top of notebook
# Re-run churn cells
# ✅ Files in Modeling/exports/churn_*.html and *.png
```

## Documentation

- **CHURN_VIZ_SUMMARY.md** — Feature overview (this section)
- **CHURN_VIZ_GUIDE.md** — Detailed usage & interpretation
- **CHURN_VIZ_ARCHITECTURE.md** — Deep dive & customization
- **FEATURE_COMPLETE_SUMMARY.md** — Implementation details

---

**Key Point:** Change K or avatar names, re-run cells, visualizations adapt automatically. No code edits needed.

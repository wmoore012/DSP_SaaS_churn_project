# 🚀 Quick Start – Churn Visualizations

## Before You Start

1. **Data file in place**

   - Make sure your engineered dataset exists at:  
     `Modeling/Dataset/DSBA_6276_model4_dataset.csv` (or `.parquet`)

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt

This includes Plotly + Kaleido for static image export.
	3.	Set export flag (optional but recommended)
At the top of the notebook:

EXPORT_SLIDES = True  # saves PNG + HTML for slides



⸻

Open & Run

1. File: Modeling/Clustering_Marketing_Avatars.ipynb
2. Section: 12A – Churn Visualization (below the PCA chart)
3. Run these 3 cells in order


⸻

What You’ll See

✅ Cell 1: Individual Churn Charts (per Cluster)
	•	One bar chart per cluster (3 or 4 depending on K)
	•	Blue bars = customers who stayed
	•	Red bars = customers who left
	•	Hover to see exact counts + churn rate

✅ Cell 2: Combined Dashboard
	•	All clusters in one view
	•	Subplots auto-layout (1×3 for K=3, 2×2 for K=4, etc.)
	•	Compare retention profiles side-by-side

✅ Cell 3: Diagnostics
	•	Automated anomaly checks:
	•	🔴 High churn warning – segment is leaking badly
	•	🟡 Tiny cluster warning – too small for big bets
	•	⚠️ Mixed signals warning – noisy or unstable group

⸻

Optional: Personas & Plan Tables (New)

After churn charts, run the persona / feature cells:
	•	Cluster persona bars
	•	Files: cluster_0_persona.png, cluster_1_persona.png, …
	•	Shows top features (z-scores) that make each avatar different from the overall base.
	•	Plan vs population chart
	•	File: cluster_plan_vs_population.png
	•	Highlights how each cluster differs on:
	•	payment_plan_days
	•	is_cancel
	•	days_to_expire

These visuals tie directly to the business story:
	•	Who’s on short vs long plans
	•	Who actively cancels
	•	Who lives right at renewal time

⸻

Customize

Change from 3 to 4 clusters

MANUAL_K = 4  # e.g., change from 3
# Re-run K-selection cell
# Re-run churn + persona visualization cells
# ✅ Now shows 4 clusters everywhere

Change avatar names (marketing labels)

MANUAL_AVATAR_OVERRIDES = {
    2: "💎 Long-Plan Locked-In Loyalists",
    1: "🔥 Veteran Power Fans",
    0: "⚠️ Wavering Short-Plan Monthlies",
    3: "🚨 Trial & Promo Burnouts",
}
# Re-run avatar assignment cell
# Re-run churn + persona cells
# ✅ Titles & legends update automatically


⸻

Export for Presentations

EXPORT_SLIDES = True  # Set at top of notebook
# Re-run churn + persona cells
# ✅ Files in Modeling/exports/*.html and *.png

Use the PNGs directly in slides; use the HTML files for live, interactive demos.

⸻

Quick Troubleshooting
	•	Error: dataset not found
→ Check Modeling/Dataset/ and the filename:
DSBA_6276_model4_dataset.csv or .parquet.
	•	Charts show only 3 clusters
→ Check MANUAL_K and re-run the K-selection + clustering cells.
	•	Avatars look wrong / old names
→ Update MANUAL_AVATAR_OVERRIDES and re-run the avatar + viz cells.

⸻

Key Point: Once data is in place, you can change K or avatar names, re-run a small set of cells, and all the churn + persona visualizations adapt automatically—no code edits needed.

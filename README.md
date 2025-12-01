# DSP_SaaS_churn_project

Graduate data science project using the KKBox music streaming subscription dataset
to engineer user behavior features, predict churn, and build machine learning
models — including clustering subscribers into marketing "avatars" — for a
DSP-style SaaS dashboard.

**Team Members:** Austin Edwards, Will Moore, Michael Johnson, Isabella Ijiogbe, Yvanna Tchomnou Chomte

This repo is a cleaned-up version of a group project for DSBA 6276.

## Project Structure

- `notebooks/` – Feature engineering, clustering, and modeling notebooks  
- `src/` – Reusable Python modules (feature engineering, K-selection utilities)  
- `data/` – Local data folder (not tracked in git)  
- `models/` – Saved models (not tracked)  
- `exports/` – Plots, tables, CSVs (not tracked)

## Data

This project uses the public KKBox churn prediction dataset from Kaggle (~130MB).  
Data is **not included** in this repository.

- Download the full dataset from  
  [Kaggle: KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge).
- If you want to reproduce the **end‑to‑end pipeline** (from raw → engineered features), place the raw CSV files under `data/raw/` and run the feature‑engineering notebook(s).
- For the **clustering notebook used in the slides**, we work from a single **feature‑engineered modeling dataset**:
  - expected filename: `DSBA_6276_model4_dataset.csv` or `DSBA_6276_model4_dataset.parquet`
  - expected location: `Modeling/Dataset/`
  - this file is not committed to git; you must either create it by running the upstream feature‑engineering notebook or copy it in from your own environment.

## Setup

```bash
# Clone the repository
gh repo clone wmoore012/DSP_SaaS_churn_project
# or:
git clone git@github.com:wmoore012/DSP_SaaS_churn_project.git

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place raw KKBox data files in `data/raw/` (if you want to rebuild from scratch).
2. (Optional) Run feature‑engineering and data‑prep notebooks:
   - `01_Feature_Engineering.ipynb`
   - `06_Cleaned_Datasets.ipynb`
   - `07_Creating_Modeling_Data.ipynb`
3. Ensure the engineered modeling dataset exists at `Modeling/Dataset/DSBA_6276_model4_dataset.csv` (or `.parquet`).
4. Run the clustering and modeling notebooks:
   - `02_Clustering_Marketing_Avatars.ipynb`  *(main notebook for the marketing avatars and plots in this README)*  
   - `03_EDA_Project.ipynb`
   - `04_Churn_Project.ipynb`
   - `05_Churn_Project_Revised.ipynb`

## Visualizations

View the static and interactive charts from our clustering analysis:

### Static Charts (PNG)
Browse key PNGs directly in the repository under `artifacts/charts/static/`. Highlights include:

- [Avatar Average Actual Amount Paid](artifacts/charts/static/avatar_avg_actual_amount_paid.png)
- [Avatar Churn Rate](artifacts/charts/static/avatar_churn_rate.png)
- [Avatar Management Playbook](artifacts/charts/static/avatar_management_playbook.png)
- [Avatar Radar](artifacts/charts/static/avatar_radar.png)
- [Churn by Avatar](artifacts/charts/static/churn_by_avatar.png)
- [Cluster Preview PCA](artifacts/charts/static/cluster_preview_pca.png)
- [K Selection Elbow](artifacts/charts/static/k_selection_elbow.png)
- [K Selection Scorecard](artifacts/charts/static/k_selection_scorecard.png)
- [PCA Clusters](artifacts/charts/static/pca_clusters.png)
- [PCA Loadings – Top Features](artifacts/charts/static/pca_loadings.png)
- [Cluster Plan vs Population](artifacts/charts/static/cluster_plan_vs_population.png)
- Cluster‑level persona bars (e.g., `cluster_0_persona.png`, `cluster_1_persona.png`, …)
- Stability diagnostics (e.g., `stability_ari_bar.png`, `stability_ari_matrix.png`, `stability_summary_table.png`)
- Z‑score heatmaps for feature differences (e.g., `zscore_heatmap_filtered.png`)

Additional static charts are generated as the notebook evolves; browse the full gallery in `artifacts/charts/static/`.

### Interactive Charts (HTML)
Hosted on GitHub Pages for full interactivity:

- [View All Interactive Charts](https://wmoore012.github.io/DSP_SaaS_churn_project/)

The gallery includes, among others:

- Interactive avatar radar and management playbook
- K‑selection scorecard and elbow curves
- PCA cluster scatter plots
- Cluster‑level persona bar charts
- Plan‑vs‑population comparisons
- Stability and z‑score diagnostics

New charts from the notebooks are automatically added to this gallery as they are exported.
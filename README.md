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

- Our group uses a shared Google Drive folder for the raw files.
- To reproduce, download the dataset from [Kaggle: KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) and place it under `data/raw/`

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

1. Place raw KKBox data files in `data/raw/`
2. Run notebooks in order:
   - `01_Feature_Engineering.ipynb`
   - `02_Clustering_Marketing_Avatars.ipynb`
   - `03_EDA_Project.ipynb`
   - `04_Churn_Project.ipynb`
   - `05_Churn_Project_Revised.ipynb`
   - `06_Cleaned_Datasets.ipynb`
   - `07_Creating_Modeling_Data.ipynb`

## Visualizations

View the static and interactive charts from our clustering analysis:

### Static Charts (PNG)
Browse directly in the repository under `artifacts/charts/static/`:
- [Avatar Average Actual Amount Paid](artifacts/charts/static/avatar_avg_actual_amount_paid.png)
- [Avatar Churn Rate](artifacts/charts/static/avatar_churn_rate.png)
- [Avatar Management Playbook](artifacts/charts/static/avatar_management_playbook.png)
- [Avatar Radar](artifacts/charts/static/avatar_radar.png)
- [Churn by Avatar](artifacts/charts/static/churn_by_avatar.png)
- [Cluster Preview PCA](artifacts/charts/static/cluster_preview_pca.png)
- [K Selection Elbow](artifacts/charts/static/k_selection_elbow.png)
- [K Selection Scorecard](artifacts/charts/static/k_selection_scorecard.png)
- [PCA Avatar Scatter](artifacts/charts/static/pca_avatar_scatter.png)
- [PCA Clusters](artifacts/charts/static/pca_clusters.png)
- [Stability ARI Bar](artifacts/charts/static/stability_ari_bar.png)
- [Stability ARI Matrix](artifacts/charts/static/stability_ari_matrix.png)
- [Stability Summary Table](artifacts/charts/static/stability_summary_table.png)
- [Z-Score Heatmap Filtered](artifacts/charts/static/zscore_heatmap_filtered.png)

### Interactive Charts (HTML)
Hosted on GitHub Pages for full interactivity:
- [View All Interactive Charts](https://wmoore012.github.io/DSP_SaaS_churn_project/)
- Individual charts: [Avatar Radar](https://wmoore012.github.io/DSP_SaaS_churn_project/charts/avatar_radar.html), [PCA Avatar Scatter](https://wmoore012.github.io/DSP_SaaS_churn_project/charts/pca_avatar_scatter.html), etc.
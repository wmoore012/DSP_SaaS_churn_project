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
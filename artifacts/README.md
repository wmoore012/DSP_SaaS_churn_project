# Artifacts

HTML and PNG deliverables generated from the notebooks live under this folder so the team can review charts without re-running the pipeline. The clustering notebook saves:

- `charts/interactive/*.html` – Plotly exports that stay under Git's 100 MB limit.
- `charts/static/*.png` – Optional slide-ready renders when `EXPORT_SLIDES=True`.

Regenerate these files by running the export cells near the end of `notebooks/02_Clustering_Marketing_Avatars.ipynb`. They are intentionally versioned, so do not add this folder to `.gitignore`.

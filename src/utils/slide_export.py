"""Slide export and archiving helpers.

This module centralizes saving of static (Matplotlib PNG) and interactive (Plotly HTML)
slides into the artifacts/charts directory, and provides an archiving routine that moves
existing slides into a timestamped archive folder prior to exporting new ones.

Typical usage in a notebook:

    from src.utils.slide_export import (
        init_slide_dirs, save_static_png_matplotlib, save_plotly_interactive_html,
        save_plotly_png, archive_existing_slides
    )

    # One-time init (no-op if dirs already exist)
    init_slide_dirs()

    # Archive any previously generated slides (run as the very last step before final exports)
    archive_existing_slides(EXPORT_SLIDES=EXPORT_SLIDES)

    # Save new figures
    save_static_png_matplotlib('churn_by_avatar.png', fig=my_matplotlib_fig, EXPORT_SLIDES=EXPORT_SLIDES)
    save_plotly_interactive_html(fig_pca, 'pca_clusters.html', EXPORT_SLIDES=EXPORT_SLIDES)

"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import shutil
import os
from typing import Optional

# --- Directory resolution ---
# Resolve project root as the directory two levels up from this file (src/utils/ -> project root)
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS_BASE = _DEFAULT_PROJECT_ROOT / 'artifacts' / 'charts'
_STATIC_SLIDES_DIR = _ARTIFACTS_BASE / 'static'
_INTERACTIVE_SLIDES_DIR = _ARTIFACTS_BASE / 'interactive'
_STATIC_ARCHIVE_DIR = _STATIC_SLIDES_DIR / 'archive'
_INTERACTIVE_ARCHIVE_DIR = _INTERACTIVE_SLIDES_DIR / 'archive'


def get_slide_dirs(project_root: Optional[Path] = None) -> dict:
    pr = Path(project_root) if project_root else _DEFAULT_PROJECT_ROOT
    artifacts = pr / 'artifacts' / 'charts'
    return {
        'project_root': pr,
        'artifacts_base': artifacts,
        'static_dir': artifacts / 'static',
        'interactive_dir': artifacts / 'interactive',
        'static_archive': artifacts / 'static' / 'archive',
        'interactive_archive': artifacts / 'interactive' / 'archive',
    }


def init_slide_dirs(project_root: Optional[Path] = None) -> dict:
    """Ensure slide directories exist. Returns a dict of important paths."""
    dirs = get_slide_dirs(project_root)
    for key in ['artifacts_base', 'static_dir', 'interactive_dir', 'static_archive', 'interactive_archive']:
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


# --- Save helpers ---

def save_static_png_matplotlib(filename_png: str, fig=None, dpi: int = 300, EXPORT_SLIDES: bool = True,
                               project_root: Optional[Path] = None):
    """Save current Matplotlib figure (or provided fig) as PNG into static slides dir."""
    if not EXPORT_SLIDES:
        print("⏭️ EXPORT_SLIDES=False, skipping PNG export →", filename_png)
        return None
    from matplotlib import pyplot as plt
    dirs = init_slide_dirs(project_root)
    path = dirs['static_dir'] / filename_png
    os.makedirs(path.parent, exist_ok=True)
    if fig is None:
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved static PNG: {path}")
    return path


def save_plotly_interactive_html(fig, filename_html: str, EXPORT_SLIDES: bool = True,
                                 project_root: Optional[Path] = None):
    """Save Plotly figure as interactive HTML into interactive slides dir."""
    if not EXPORT_SLIDES:
        print("⏭️ EXPORT_SLIDES=False, skipping HTML export →", filename_html)
        return None
    dirs = init_slide_dirs(project_root)
    path = dirs['interactive_dir'] / filename_html
    os.makedirs(path.parent, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs='cdn', full_html=True)
    print(f"✅ Saved interactive HTML: {path}")
    return path


def save_plotly_png(fig, filename_png: str, scale: int = 2, EXPORT_SLIDES: bool = True,
                    project_root: Optional[Path] = None):
    """Save Plotly figure as static PNG via kaleido (optional dependency)."""
    if not EXPORT_SLIDES:
        print("⏭️ EXPORT_SLIDES=False, skipping Plotly PNG export →", filename_png)
        return None
    dirs = init_slide_dirs(project_root)
    path = dirs['static_dir'] / filename_png
    os.makedirs(path.parent, exist_ok=True)
    try:
        fig.write_image(str(path), scale=scale)  # requires `pip install -U kaleido`
        print(f"✅ Saved plotly PNG: {path}")
        return path
    except Exception as e:
        print("⚠️ Could not export Plotly PNG via kaleido:", e)
        return None


# --- Archiving ---

def archive_existing_slides(EXPORT_SLIDES: bool = True, project_root: Optional[Path] = None):
    """Move current slides into timestamped archive folders under static/archive and interactive/archive.
    Skips if EXPORT_SLIDES is False. Returns a dict with archive destination paths or None if skipped.
    """
    if not EXPORT_SLIDES:
        print("⏭️ EXPORT_SLIDES=False, skipping archive step")
        return {"static": None, "interactive": None}

    dirs = init_slide_dirs(project_root)
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary: dict[str, Optional[Path]] = {}

    def _archive_one(slides_dir: Path, archive_dir: Path, label: str):
        entries = [p for p in slides_dir.iterdir() if p.name != 'archive']
        if not entries:
            print(f"ℹ️ No {label} slides to archive in {slides_dir}")
            summary[label] = None
            return
        dest = archive_dir / ts
        dest.mkdir(parents=True, exist_ok=True)
        moved = 0
        for p in entries:
            target = dest / p.name
            try:
                shutil.move(str(p), str(target))
                moved += 1
            except Exception as e:
                print(f"⚠️ Failed to archive {p} → {target}: {e}")
        print(f"📦 Archived {moved} {label} item(s) to {dest}")
        summary[label] = dest

    _archive_one(dirs['static_dir'], dirs['static_archive'], 'static')
    _archive_one(dirs['interactive_dir'], dirs['interactive_archive'], 'interactive')

    return summary

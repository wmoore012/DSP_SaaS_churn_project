"""Matplotlib readability helpers and tick formatters.

Provides:
- apply_adaptive_labels(ax): adapt tick density and label rotation/size
- formatters: CURRENCY_FMT, PERCENT_FMT, THOUSANDS_FMT
- optional adjustText integration for label overlap avoidance
"""
from __future__ import annotations

from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

try:
    from adjustText import adjust_text  # re-export convenience
except Exception:  # pragma: no cover
    adjust_text = None  # type: ignore


def apply_adaptive_labels(ax):
    """Adapt x/y tick density and label sizing/rotation for readability."""
    num_ticks = len(ax.get_xticklabels())
    if num_ticks > 10:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    elif num_ticks > 5:
        plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    else:
        plt.setp(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


CURRENCY_FMT = plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
PERCENT_FMT  = plt.FuncFormatter(lambda x, p: f'{x:.0%}')
THOUSANDS_FMT= plt.FuncFormatter(lambda x, p: f'{x:,.0f}')

"""
K-Selection utility for clustering workflows.

Implements a two-phase approach:
1. Phase 1 (Elbow): Fast inertia-only scoring across a K range
2. Phase 2 (Silhouette): Refined scoring on candidate Ks only

This avoids expensive O(n²) silhouette computations on obviously bad K values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringConfig:
    """Configuration for K selection and clustering."""

    k_range: Tuple[int, int] = (2, 9)  # Tests K = 2..8 (range end is exclusive)
    sample_size: int = 60_000  # Max sample for elbow/silhouette (lowered from 150k for speed)
    silhouette_subsample: int = 20_000  # Subsample for silhouette to avoid O(n²) on large n
    batch_size: int = 4096
    random_state: int = 42
    min_cluster_pct: float = 5.0  # Min % of users in any cluster
    candidate_window: int = 3  # Number of Ks around elbow (e.g., 3 → elbow ±1)
    candidate_ks: Optional[List[int]] = None  # Manual override for candidate Ks


class KSelector:
    """Two-phase K selector: elbow (inertia) then silhouette-based refinement."""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.elbow_df: Optional[pd.DataFrame] = None
        self.metrics_df: Optional[pd.DataFrame] = None
        self.last_elbow_k: Optional[int] = None

    def _sample(self, X: np.ndarray, size: int) -> np.ndarray:
        """Return a reproducible random subset of rows."""
        size = min(size, X.shape[0])
        rng = np.random.default_rng(self.config.random_state)
        idx = rng.choice(X.shape[0], size=size, replace=False)
        return X[idx]

    def _auto_candidate_ks_from_elbow(self) -> List[int]:
        """
        Derive candidate Ks around the elbow if none were manually provided.

        Finds the K with the largest relative inertia drop and builds a window around it.
        """
        if self.elbow_df is None:
            raise ValueError("run_elbow_only() must be called before auto-deriving candidate Ks.")

        df = self.elbow_df.sort_values("k").reset_index(drop=True)

        # Compute relative inertia drop between successive K values
        df["inertia_diff"] = df["inertia"].shift(1) - df["inertia"]
        df["rel_drop"] = df["inertia_diff"] / (df["inertia"].shift(1) + 1e-9)  # Avoid div/0 at K=2

        # Find K with largest relative drop (the "elbow")
        elbow_row = df.loc[df["rel_drop"].idxmax()]
        elbow_k = int(elbow_row["k"])
        self.last_elbow_k = elbow_k

        # Build a window of candidate_window size around elbow_k
        half_window = self.config.candidate_window // 2
        k_min, k_max = self.config.k_range
        candidates = [
            k
            for k in range(elbow_k - half_window, elbow_k + half_window + 1)
            if k_min <= k < k_max
        ]
        return sorted(set(candidates))

    def run_elbow_only(self, X: np.ndarray, verbose: bool = True) -> pd.DataFrame:
        """
        Phase 1: Run MiniBatchKMeans for K in k_range and record inertia only.

        This is fast (O(n)) and provides data for the elbow plot (K vs inertia).
        """
        X_sample = self._sample(X, self.config.sample_size)
        k_start, k_stop = self.config.k_range

        rows = []
        if verbose:
            print("🔍 Phase 1: Elbow Method (Inertia Only)")
            print(f"   Sample size: {X_sample.shape[0]:,} rows")
            print(f"   K range: {k_start} to {k_stop - 1}")
            print("-" * 50)

        for k in range(k_start, k_stop):
            if verbose:
                print(f"  K={k}...", end="", flush=True)

            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=self.config.random_state,
                n_init="auto",
                batch_size=self.config.batch_size,
            )
            km.fit(X_sample)
            inertia = km.inertia_
            rows.append({"k": k, "inertia": inertia})

            if verbose:
                print(f" inertia={inertia:,.0f}")

        self.elbow_df = pd.DataFrame(rows)

        if verbose:
            print("✅ Phase 1 complete\n")

        return self.elbow_df

    def run_selection(self, X: np.ndarray, verbose: bool = True) -> pd.DataFrame:
        """
        Phase 2: Score candidate Ks using silhouette + min_cluster_pct constraint.

        - If config.candidate_ks is set manually, uses those values
        - Otherwise, derives candidates automatically from the elbow
        """
        # Determine candidate Ks
        if self.config.candidate_ks is not None:
            candidate_ks = sorted(self.config.candidate_ks)
            elbow_k = None
            if verbose:
                print(f"🎯 Using manually specified candidate Ks: {candidate_ks}")
        else:
            candidate_ks = self._auto_candidate_ks_from_elbow()
            elbow_k = self.last_elbow_k
            if verbose:
                print(f"   Auto-detected elbow at K={elbow_k}")
                print(f"🎯 Auto-derived candidate Ks from elbow: {candidate_ks}")

        if verbose:
            print("\n🔬 Phase 2: Silhouette Scoring")
            print(f"   Candidates: {candidate_ks}")
            print(f"   Sample size: {self.config.sample_size:,} rows")
            print(f"   Silhouette subsample: {self.config.silhouette_subsample:,} rows")
            print("-" * 50)

        # Sample once for all candidates
        X_sample = self._sample(X, self.config.sample_size)

        rows = []
        for i, k in enumerate(candidate_ks, start=1):
            if verbose:
                print(f"  [{i}/{len(candidate_ks)}] K={k}...", end=" ", flush=True)

            # Fit model on full sample
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=self.config.random_state,
                n_init="auto",
                batch_size=self.config.batch_size,
            )
            labels_full = km.fit_predict(X_sample)

            # Check minimum cluster size constraint
            _, counts = np.unique(labels_full, return_counts=True)
            min_pct = counts.min() / len(labels_full) * 100

            # Compute silhouette on subsample for speed
            rng = np.random.default_rng(self.config.random_state + k)
            sil_size = min(self.config.silhouette_subsample, X_sample.shape[0])
            idx_sil = rng.choice(X_sample.shape[0], size=sil_size, replace=False)
            X_sil = X_sample[idx_sil]
            labels_sil = labels_full[idx_sil]
            sil = silhouette_score(X_sil, labels_sil)

            row = {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": sil,
                "min_cluster_pct": min_pct,
            }
            rows.append(row)

            # Status indicator
            status = "⚠️" if min_pct < self.config.min_cluster_pct else "✓"
            if verbose:
                print(f"sil={sil:.3f}, min_cluster={min_pct:.1f}% {status}")

        self.metrics_df = pd.DataFrame(rows)

        if verbose:
            print("\n✅ Phase 2 complete\n")

        return self.metrics_df

    def recommend_k(self) -> int:
        """
        Select the best K from Phase 2 results.

        Chooses the K with highest silhouette score among those meeting
        the min_cluster_pct constraint.
        """
        if self.metrics_df is None:
            raise ValueError("run_selection() must be called before recommend_k().")

        df = self.metrics_df.copy()
        valid = df[df["min_cluster_pct"] >= self.config.min_cluster_pct]

        if valid.empty:
            print("⚠️ No K satisfies min_cluster_pct constraint. Using best silhouette anyway.")
            valid = df

        best_row = valid.loc[valid["silhouette"].idxmax()]
        return int(best_row["k"])

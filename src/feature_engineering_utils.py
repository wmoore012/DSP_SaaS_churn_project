"""
Feature Engineering Utilities for KKBox Churn Analysis
======================================================

This module extracts the EXACT feature engineering logic from Feature Engineering.ipynb
into reusable functions. No new features are invented here - this is a clean refactor
of the team's existing work for reproducibility and reuse.

Usage:
    from feature_engineering_utils import apply_feature_engineering
    df_engineered = apply_feature_engineering(df_raw)

Author: [Your Name]
Date: November 2025
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def apply_feature_engineering(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline from Feature Engineering.ipynb.
    
    This function reproduces ALL transformations from the team's notebook:
    - Temporal features (account age, renewal gaps, recency flags)
    - Transaction features (payment ratios, renewal consistency)
    - Engagement features (skip ratio, completion ratio, engagement intensity)
    - Demographic features (age cleaning, city encoding)
    - Derived interaction features (value_for_money, etc.)
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw model dataset loaded from DSBA_6276_model_dataset
    verbose : bool
        If True, print progress messages
        
    Returns
    -------
    pd.DataFrame
        Fully engineered dataframe ready for modeling/clustering
        
    Notes
    -----
    This function modifies a COPY of the input dataframe.
    Original data is not mutated.
    """
    if verbose:
        print("🔧 Starting feature engineering pipeline...")
        print(f"   Input shape: {df.shape}")
    
    # Work on a copy to avoid side effects
    df = df.copy()
    
    # Step 1: Temporal Features
    if verbose:
        print("   [1/6] Engineering temporal features...")
    df = _engineer_temporal_features(df)
    
    # Step 2: Transaction Features  
    if verbose:
        print("   [2/6] Engineering transaction features...")
    df = _engineer_transaction_features(df)
    
    # Step 3: Engagement Features
    if verbose:
        print("   [3/6] Engineering engagement features...")
    df = _engineer_engagement_features(df)
    
    # Step 4: Demographic Features
    if verbose:
        print("   [4/6] Engineering demographic features...")
    df = _engineer_demographic_features(df)
    
    # Step 5: Derived & Interaction Features
    if verbose:
        print("   [5/6] Engineering derived/interaction features...")
    df = _engineer_derived_features(df)
    
    # Step 6: Final cleanup
    if verbose:
        print("   [6/6] Final cleanup (inf/NaN handling)...")
    df = _final_cleanup(df)
    
    if verbose:
        print(f"✅ Feature engineering complete! Output shape: {df.shape}")
        
    return df


# =============================================================================
# TEMPORAL FEATURES (from Feature Engineering.ipynb cells 10-11)
# =============================================================================

def _engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal features: account age, renewal gaps, recency flags.
    Exact logic from Feature Engineering.ipynb.
    """
    # Gracefully handle missing activity date column by falling back to the
    # transaction date so downstream recency features stay defined.
    has_activity_date = "date" in df.columns

    # Ensure datetime dtypes
    date_cols = ["registration_init_time", "transaction_date", "membership_expire_date", "date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if not has_activity_date:
        df["date"] = df["transaction_date"] if "transaction_date" in df.columns else pd.NaT
    
    # Compute per-user renewal gaps (days between consecutive transactions)
    df_sorted = df.sort_values(["msno", "transaction_date"])
    df_sorted["renewal_gap_days"] = (
        df_sorted.groupby("msno")["transaction_date"].diff().dt.days
    )
    
    # Aggregate to user-level temporal table
    temporal = df_sorted.groupby("msno").agg(
        registration_init_time=("registration_init_time", "first"),
        last_transaction_date=("transaction_date", "last"),
        last_expire_date=("membership_expire_date", "last"),
        last_activity_date=("date", "max"),
        mean_renewal_gap_days=("renewal_gap_days", "mean"),
        last_plan_days=("payment_plan_days", "last"),
        num_transactions=("num_transactions", "max")
    ).reset_index()
    
    # Derive temporal features
    temporal["account_age_days"] = (
        temporal["last_transaction_date"] - temporal["registration_init_time"]
    ).dt.days
    
    temporal["plan_duration_days_last"] = (
        temporal["last_expire_date"] - temporal["last_transaction_date"]
    ).dt.days
    
    temporal["days_until_expiration_from_activity"] = (
        temporal["last_expire_date"] - temporal["last_activity_date"]
    ).dt.days
    
    temporal["days_since_last_activity_to_tx"] = (
        temporal["last_transaction_date"] - temporal["last_activity_date"]
    ).dt.days
    
    # Cohort features
    temporal["registration_year"] = temporal["registration_init_time"].dt.year
    temporal["registration_month"] = temporal["registration_init_time"].dt.month
    
    # Recency flags
    temporal["is_recent_user_90d"] = (temporal["account_age_days"] <= 90).astype(int)
    temporal["is_long_term_user_365d"] = (temporal["account_age_days"] >= 365).astype(int)
    
    # Weekend signal
    temporal["last_activity_is_weekend"] = (
        temporal["last_activity_date"].dt.weekday.isin([5, 6]).astype(int)
    )
    
    # Handle inf/NaN
    temporal_numeric_cols = [
        "mean_renewal_gap_days", "plan_duration_days_last",
        "days_until_expiration_from_activity", "days_since_last_activity_to_tx",
        "account_age_days"
    ]
    for c in temporal_numeric_cols:
        if c in temporal.columns:
            temporal[c] = temporal[c].replace([np.inf, -np.inf], np.nan)
    
    temporal = temporal.fillna({
        "mean_renewal_gap_days": 0,
        "plan_duration_days_last": 0,
        "days_until_expiration_from_activity": 0,
        "days_since_last_activity_to_tx": 0,
        "account_age_days": 0,
    })
    
    # Merge back
    df = df.merge(temporal, on="msno", how="left", suffixes=("", "_temporal"))
    
    return df


# =============================================================================
# TRANSACTION FEATURES (from Feature Engineering.ipynb cells 12-13)
# =============================================================================

def _engineer_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer transaction-based features: payment ratios, renewal consistency.
    Exact logic from Feature Engineering.ipynb.
    """
    tx_cols = [
        "msno", "transaction_date", "membership_expire_date",
        "payment_plan_days", "plan_list_price", "actual_amount_paid",
        "payment_method_id", "is_auto_renew", "is_cancel"
    ]
    tx = df.loc[:, [c for c in tx_cols if c in df.columns]].dropna(subset=["msno"]).copy()
    
    # Sort for per-user computations
    tx = tx.sort_values(["msno", "transaction_date"])
    
    # Renewal gaps
    tx["renewal_gap_days"] = tx.groupby("msno")["transaction_date"].diff().dt.days
    
    # Discount signal
    if {"actual_amount_paid", "plan_list_price"}.issubset(tx.columns):
        tx["discount_log_diff"] = tx["actual_amount_paid"] - tx["plan_list_price"]
    
    # Transaction count per user
    tx_count = tx.groupby("msno").size().rename("tx_count")
    
    # Aggregations
    tx_agg = tx.groupby("msno").agg(
        total_payment=("actual_amount_paid", "sum"),
        avg_payment=("actual_amount_paid", "mean"),
        payment_std=("actual_amount_paid", "std"),
        total_plan_days=("payment_plan_days", "sum"),
        avg_plan_days=("payment_plan_days", "mean"),
        max_plan_days=("payment_plan_days", "max"),
        num_plan_changes=("payment_plan_days", pd.Series.nunique),
        num_payment_methods=("payment_method_id", pd.Series.nunique),
        avg_renewal_gap_days=("renewal_gap_days", "mean"),
        renewal_gap_std=("renewal_gap_days", "std"),
        last_auto_renew=("is_auto_renew", lambda x: x.iloc[-1] if len(x) > 0 else 0),
        last_cancel=("is_cancel", lambda x: x.iloc[-1] if len(x) > 0 else 0),
        mean_discount_log_diff=("discount_log_diff", "mean"),
        last_discount_log_diff=("discount_log_diff", lambda x: x.iloc[-1] if len(x) > 0 else 0),
    )
    
    tx_feats = tx_agg.join(tx_count, how="left").reset_index()
    
    # Derived ratios with safe division
    tx_feats["payment_per_day"] = np.where(
        tx_feats["total_plan_days"] > 0,
        tx_feats["total_payment"] / tx_feats["total_plan_days"],
        0.0
    )
    
    tx_feats["payment_per_transaction"] = np.where(
        tx_feats["tx_count"] > 0,
        tx_feats["total_payment"] / tx_feats["tx_count"],
        0.0
    )
    
    # Renewal consistency (inverse of gap variability)
    tx_feats["renewal_consistency"] = 1.0 / (1.0 + tx_feats["renewal_gap_std"].fillna(0.0))
    
    # Fill NaNs
    tx_feats = tx_feats.fillna({
        "payment_std": 0.0,
        "avg_renewal_gap_days": 0.0,
        "renewal_gap_std": 0.0,
        "mean_discount_log_diff": 0.0,
        "last_discount_log_diff": 0.0,
    })
    
    # Merge back
    df = df.merge(tx_feats, on="msno", how="left", suffixes=("", "_tx"))
    
    return df


# =============================================================================
# ENGAGEMENT FEATURES (from Feature Engineering.ipynb cells 14-15)
# =============================================================================

def _engineer_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer user engagement features: listening behavior ratios.
    Exact logic from Feature Engineering.ipynb.
    """
    eng_cols = [
        "msno", "num_25", "num_50", "num_75", "num_985", "num_100",
        "num_unq", "total_secs"
    ]
    eng_df = df.loc[:, [c for c in eng_cols if c in df.columns]].copy()
    
    # Aggregate listening activity per user
    eng_agg = eng_df.groupby("msno").agg(
        num_25_sum=("num_25", "sum"),
        num_50_sum=("num_50", "sum"),
        num_75_sum=("num_75", "sum"),
        num_985_sum=("num_985", "sum"),
        num_100_sum=("num_100", "sum"),
        num_unq_mean=("num_unq", "mean"),
        total_secs_sum=("total_secs", "sum"),
        total_secs_mean=("total_secs", "mean")
    ).reset_index()
    
    # Skip ratio: early skips vs. longer listens
    eng_agg["skip_ratio"] = (
        (eng_agg["num_25_sum"] + eng_agg["num_50_sum"]) /
        (eng_agg["num_75_sum"] + eng_agg["num_985_sum"] + 1)
    )
    
    # Completion ratio: finished songs
    eng_agg["completion_ratio"] = (
        (eng_agg["num_985_sum"] + eng_agg["num_100_sum"]) /
        (eng_agg["num_25_sum"] + eng_agg["num_50_sum"] + 1)
    )
    
    # Engagement ratio: listening intensity
    eng_agg["engagement_ratio"] = (
        eng_agg["total_secs_sum"] / (eng_agg["num_unq_mean"] + 1)
    )
    
    # Unique song ratio: diversity
    eng_agg["unique_song_ratio"] = (
        eng_agg["num_unq_mean"] / (
            eng_agg["num_25_sum"] + eng_agg["num_50_sum"] +
            eng_agg["num_75_sum"] + eng_agg["num_985_sum"] + 1
        )
    )
    
    # Handle inf/NaN
    eng_agg = eng_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Merge back
    df = df.merge(eng_agg, on="msno", how="left", suffixes=("", "_eng"))
    
    return df


# =============================================================================
# DEMOGRAPHIC FEATURES (from Feature Engineering.ipynb cells 16-21)
# =============================================================================

def _engineer_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer demographic features: age, gender, city, registration channel.
    Exact logic from Feature Engineering.ipynb.
    """
    # 1) Age (bd) cleaning and bucketing
    if "bd" in df.columns:
        df["bd"] = df["bd"].where(df["bd"].between(5, 100), np.nan)
        bd_median = df["bd"].median(skipna=True)
        df["bd_imputed"] = df["bd"].fillna(bd_median if pd.notna(bd_median) else 25)
        
        # Age buckets
        bins = [0, 17, 24, 34, 44, 54, 64, 200]
        labels = ["<=17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        df["bd_bucket"] = pd.cut(
            df["bd_imputed"], bins=bins, labels=labels, 
            right=True, include_lowest=True
        )
    
    # 2) Gender one-hot (already encoded: male=0, female=1, unknown=2)
    if "gender" in df.columns:
        gender_dummies = pd.get_dummies(
            df["gender"].astype("category"), prefix="gender", drop_first=False
        )
        df = pd.concat([df, gender_dummies], axis=1)
    
    # 3) City encoding (frequency encoding for high cardinality)
    if "city" in df.columns:
        city_counts = df["city"].value_counts()
        df["city_freq"] = df["city"].map(city_counts)
        df["city_freq_norm"] = df["city_freq"] / len(df)
        
        # Top-N one-hot
        TOP_N = 10
        top_cities = city_counts.head(TOP_N).index
        df["city_other"] = (~df["city"].isin(top_cities)).astype(int)
        
        city_top_dummies = pd.get_dummies(
            df.loc[df["city"].isin(top_cities), "city"], prefix="city_top"
        )
        city_top_dummies = city_top_dummies.reindex(df.index, fill_value=0)
        df = pd.concat([df, city_top_dummies], axis=1)
    
    # 4) Registration channel
    if "registered_via" in df.columns:
        regvia_dummies = pd.get_dummies(
            df["registered_via"].astype("category"), prefix="regvia", drop_first=False
        )
        df = pd.concat([df, regvia_dummies], axis=1)
    
    # 5) Registration cohorts
    if "registration_init_time" in df.columns:
        reg_time = pd.to_datetime(df["registration_init_time"], errors="coerce")
        df["reg_year"] = reg_time.dt.year
        df["reg_month"] = reg_time.dt.month
        df["reg_cohort"] = reg_time.dt.to_period("Q").astype(str)
        
        reg_year_dummies = pd.get_dummies(
            df["reg_year"].astype("Int64"), prefix="regyear", dummy_na=True
        )
        df = pd.concat([df, reg_year_dummies], axis=1)
    
    return df


# =============================================================================
# DERIVED & INTERACTION FEATURES (from Feature Engineering.ipynb cells 23-28)
# =============================================================================

def _engineer_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer derived and interaction features.
    Exact logic from Feature Engineering.ipynb.
    """
    # 1) Payment-Activity Interactions
    if {"total_secs_sum", "total_plan_days", "total_payment"}.issubset(df.columns):
        df["secs_per_plan_day"] = np.where(
            df["total_plan_days"] > 0,
            df["total_secs_sum"] / df["total_plan_days"],
            0
        )
        df["payment_per_second"] = np.where(
            df["total_secs_sum"] > 0,
            df["total_payment"] / df["total_secs_sum"],
            0
        )
    
    # 2) Listening Behavior Ratios
    if {"num_unq_mean", "total_secs_sum"}.issubset(df.columns):
        df["secs_per_unique_song"] = np.where(
            df["num_unq_mean"] > 0,
            df["total_secs_sum"] / df["num_unq_mean"],
            0
        )
    
    if {"completion_ratio", "avg_plan_days"}.issubset(df.columns):
        df["completion_per_plan_day"] = df["completion_ratio"] / (df["avg_plan_days"] + 1)
    
    # 3) Demographic-Behavior Interactions
    if {"bd_imputed", "engagement_ratio"}.issubset(df.columns):
        df["age_x_engagement"] = df["bd_imputed"] * df["engagement_ratio"]
    
    if {"gender", "completion_ratio"}.issubset(df.columns):
        df["gender_x_completion"] = df["gender"] * df["completion_ratio"]
    
    # 4) Renewal Consistency & Auto-Renew Interactions
    if {"renewal_consistency", "last_auto_renew"}.issubset(df.columns):
        df["auto_renew_x_consistency"] = df["renewal_consistency"] * df["last_auto_renew"]
    
    # 5) Value for Money (KEY CLUSTERING FEATURE)
    if {"engagement_ratio", "payment_per_day"}.issubset(df.columns):
        df["value_for_money"] = np.where(
            df["payment_per_day"] > 0,
            df["engagement_ratio"] / df["payment_per_day"],
            0
        )
    
    # 6) Stability Features
    if {"num_plan_changes", "num_transactions"}.issubset(df.columns):
        df["plan_change_ratio"] = np.where(
            df["num_transactions"] > 0,
            df["num_plan_changes"] / df["num_transactions"],
            0
        )
    
    return df


# =============================================================================
# FINAL CLEANUP
# =============================================================================

def _final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup: replace inf/NaN in all numeric columns.
    Exact logic from Feature Engineering.ipynb.
    """
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(0)
    return df


# =============================================================================
# UTILITY FUNCTIONS FOR CLUSTERING
# =============================================================================

def get_clustering_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract features suitable for clustering from an engineered dataframe.
    
    Removes:
    - ID columns (msno)
    - Target column (is_churn) 
    - Date/object columns
    - Highly sparse one-hot columns (optional)
    
    Parameters
    ----------
    df : pd.DataFrame
        Engineered dataframe from apply_feature_engineering()
    exclude_cols : list, optional
        Additional columns to exclude
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix for clustering
    feature_names : list
        List of feature column names
    """
    # Default exclusions
    default_exclude = {"msno", "is_churn", "bd_bucket", "reg_cohort"}
    
    if exclude_cols:
        default_exclude.update(exclude_cols)
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Filter out excluded columns
    feature_cols = [c for c in numeric_cols if c not in default_exclude]
    
    X = df[feature_cols].copy()
    
    return X, feature_cols


def get_recommended_clustering_features() -> List[str]:
    """
    Return a curated list of features recommended for customer segmentation.
    
    These are the most interpretable behavioral/value features for
    creating marketing avatars.
    
    Returns
    -------
    list
        Feature names for clustering
    """
    return [
        # Engagement metrics
        "total_secs_sum",        # Total listening time
        "skip_ratio",            # Early skip behavior
        "completion_ratio",      # Song completion behavior
        "engagement_ratio",      # Listening intensity
        "num_unq_mean",          # Music variety/diversity

        # Payment/value metrics
        "payment_per_day",       # Daily spend rate
        "total_payment",         # Total revenue
        "avg_plan_days",         # Preferred plan length
        "value_for_money",       # Engagement per dollar

        # Loyalty/retention signals
        "last_auto_renew",       # Auto-renewal status
        "renewal_consistency",   # Renewal regularity
        "account_age_days",      # Customer tenure
        "tx_count",              # Number of transactions

        # Demographics (optional)
        "bd_imputed",            # Age
        "city_freq_norm",        # City population proxy
    ]


# =============================================================================
# RAW SCHEMA VALIDATION & AVATAR HELPERS
# =============================================================================


@dataclass
class RawSchemaValidationResult:
    """Result of validating a raw input schema.

    Intentionally simple: just reports present/missing/extra columns so the
    notebook can decide how hard to fail.
    """

    present_required: Set[str]
    missing_required: Set[str]
    present_optional: Set[str]
    missing_optional: Set[str]
    unexpected: Set[str]


def validate_raw_schema(
    cols: Sequence[str],
    required: Sequence[str],
    optional: Optional[Sequence[str]] = None,
) -> RawSchemaValidationResult:
    """Compare an observed column set against required/optional expectations.

    Parameters
    ----------
    cols : sequence of str
        Actual column names found in the raw dataset.
    required : sequence of str
        Columns that *must* be present for feature engineering to behave.
    optional : sequence of str, optional
        Columns that are nice-to-have but not required.
    """

    colset: Set[str] = set(cols)
    req: Set[str] = set(required)
    opt: Set[str] = set(optional or [])

    return RawSchemaValidationResult(
        present_required=req & colset,
        missing_required=req - colset,
        present_optional=opt & colset,
        missing_optional=opt - colset,
        unexpected=colset - (req | opt),
    )


def assign_avatar_names_simple(
    cluster_summary: pd.DataFrame,
    core_features_available: bool,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Attach avatar labels/colors or fall back to generic cluster names.

    This helper is intentionally conservative:

    - If ``core_features_available`` is False, it *only* assigns neutral
      "Cluster {k}" labels and a gray color. We do **not** try to guess
      rich marketing avatars without the right stats.
    - If ``core_features_available`` is True, the notebook is expected to
      apply its richer AvatarProfiler logic before or after this call.
      Here we simply return the input unchanged so behavior stays
      backwards-compatible in the "perfect data" case.
    """

    summary = cluster_summary.copy()

    if not core_features_available:
        # Neutral, schema-safe fallback: honest generic cluster labels.
        if cluster_col not in summary.columns:
            raise KeyError(
                f"cluster_summary must contain a '{cluster_col}' column for avatar assignment"
            )

        summary["avatar"] = summary[cluster_col].apply(lambda k: f"Cluster {k}")
        summary["avatar_color"] = "#999999"  # neutral gray
        if "avatar_description" not in summary.columns:
            summary["avatar_description"] = "Generic segment (core stats missing)"
        if "churn_risk_level" not in summary.columns:
            summary["churn_risk_level"] = "Unknown"

        return summary

    # When core features are available, the richer avatar logic in the
    # notebook (AvatarProfiler, etc.) should run. We simply return the
    # summary as-is to keep behavior unchanged.
    return summary

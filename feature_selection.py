"""
Feature Selection Module for InSDN Dataset

This module provides a three-tier feature selection framework:
- Tier 1: Mandatory drops (identifiers, targets, constants)
- Tier 2: Statistical filtering (variance, missing data, correlation)
- Tier 3: Predictive power selection (mutual information, f-test, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, 
    mutual_info_classif, f_classif, chi2,
    SelectFromModel, RFE
)
from sklearn.ensemble import RandomForestClassifier


def tier1_mandatory_drops(df, label_col='LabelBinary'):
    """
    TIER 1: Remove identifiers, targets, and constant columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    label_col : str
        Name of the label column to drop
        
    Returns:
    --------
    X : pd.DataFrame
        Features with mandatory columns removed
    dropped_cols : list
        List of columns that were dropped
    """
    # Columns to always drop
    drop_cols = [
        label_col,      # prediction target
        'Label',        # original label column (multi-class)
        '__source__',   # provenance
        'Flow ID',      # unique identifier
        'Src IP',       # source IP address
        'Dst IP',       # destination IP address
        'Timestamp',    # timestamp
    ]
    
    # Keep only columns that actually exist
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    
    # Remove constant columns (0 or 1 unique value)
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=const_cols)
    
    all_dropped = drop_cols + const_cols
    
    print(f"Tier 1: Dropped {len(all_dropped)} columns")
    print(f"  - Identifiers/targets: {len(drop_cols)}")
    print(f"  - Constant columns: {len(const_cols)}")
    
    return X, all_dropped


def tier2_statistical_filtering(X, y,
                                variance_threshold=0.01,
                                missing_threshold=0.5,
                                correlation_threshold=0.95):
    """
    TIER 2: Remove low-variance, high-missing, and highly correlated features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable
    variance_threshold : float
        Minimum variance to keep a feature (default: 0.01)
    missing_threshold : float
        Maximum missing ratio to keep a feature (default: 0.5)
    correlation_threshold : float
        Maximum correlation between features (default: 0.95)
        
    Returns:
    --------
    X_filtered : pd.DataFrame
        Features after statistical filtering
    stats : dict
        Statistics about what was dropped
    """
    original_count = len(X.columns)
    stats = {
        'variance_dropped': 0,
        'missing_dropped': 0,
        'correlation_dropped': 0
    }
    
    # 1. Low variance filter
    var_selector = VarianceThreshold(threshold=variance_threshold)
    X_var = var_selector.fit_transform(X)
    var_features = X.columns[var_selector.get_support()].tolist()
    X = X[var_features]
    stats['variance_dropped'] = original_count - len(var_features)
    print(f"Tier 2a: Variance filter - {len(var_features)} features remain "
          f"({stats['variance_dropped']} dropped)")
    
    # 2. High missing data filter
    missing_ratio = X.isnull().sum() / len(X)
    low_missing = missing_ratio[missing_ratio <= missing_threshold].index.tolist()
    stats['missing_dropped'] = len(X.columns) - len(low_missing)
    X = X[low_missing]
    print(f"Tier 2b: Missing data filter - {len(low_missing)} features remain "
          f"({stats['missing_dropped']} dropped)")
    
    # 3. High correlation filter
    # Calculate correlation matrix
    corr_matrix = X.select_dtypes(include=[np.number]).corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find highly correlated pairs
    to_remove = set()
    for col in upper_triangle.columns:
        high_corr = upper_triangle.index[upper_triangle[col] > correlation_threshold].tolist()
        for feat in high_corr:
            # Keep feature with higher variance
            if col in X.columns and feat in X.columns:
                if X[col].var() > X[feat].var():
                    to_remove.add(feat)
                else:
                    to_remove.add(col)
    
    stats['correlation_dropped'] = len(to_remove)
    X = X.drop(columns=[c for c in to_remove if c in X.columns])
    print(f"Tier 2c: Correlation filter - {len(X.columns)} features remain "
          f"({stats['correlation_dropped']} dropped)")
    
    return X, stats


def tier3_predictive_selection(X, y,
                              method='mutual_info',
                              n_features=30,
                              random_state=42):
    """
    TIER 3: Select top features based on predictive power.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target variable
    method : str
        Selection method: 'mutual_info', 'f_test', 'chi2', 'tree', or 'rfe'
    n_features : int
        Number of features to select
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    X_selected : pd.DataFrame
        Selected features
    feature_scores : pd.DataFrame
        Feature importance scores
    """
    if method == 'mutual_info':
        # Mutual Information - good for non-linear relationships
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        method_name = 'Mutual Information'
        
    elif method == 'f_test':
        # F-test / ANOVA - good for linear relationships
        selector = SelectKBest(score_func=f_classif, k=n_features)
        method_name = 'F-test (ANOVA)'
        
    elif method == 'chi2':
        # Chi-squared - for categorical features (requires non-negative)
        # Only apply to non-negative features
        X_nonneg = X.select_dtypes(include=[np.number])
        X_nonneg = X_nonneg - X_nonneg.min()  # Make non-negative
        selector = SelectKBest(score_func=chi2, k=n_features)
        method_name = 'Chi-squared'
        X = X_nonneg
        
    elif method == 'tree':
        # Tree-based feature importance
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state, 
            n_jobs=-1,
            max_depth=10  # Prevent overfitting
        )
        rf.fit(X, y)
        selector = SelectFromModel(rf, threshold='median')
        method_name = 'Random Forest Importance'
        
    elif method == 'rfe':
        # Recursive Feature Elimination
        rf = RandomForestClassifier(
            n_estimators=50, 
            random_state=random_state, 
            n_jobs=-1,
            max_depth=10
        )
        selector = RFE(rf, n_features_to_select=n_features, step=1)
        method_name = 'Recursive Feature Elimination'
        
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Choose from: 'mutual_info', 'f_test', 'chi2', 'tree', 'rfe'")
    
    # Fit selector
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    if hasattr(selector, 'get_support'):
        selected_features = X.columns[selector.get_support()].tolist()
    else:
        # For RFE
        selected_features = X.columns[selector.support_].tolist()
    
    # Get scores for reporting
    if hasattr(selector, 'scores_'):
        scores = selector.scores_
    elif hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
        scores = selector.estimator_.feature_importances_
    else:
        scores = np.ones(len(X.columns))  # Placeholder
    
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': scores
    }).sort_values('score', ascending=False)
    
    print(f"Tier 3: Selected {len(selected_features)} features using {method_name}")
    print(f"\nTop 10 features by score:")
    print(feature_scores.head(10)[['feature', 'score']].to_string(index=False))
    
    return X[selected_features], feature_scores


def feature_selection_pipeline(df, y,
                              variance_threshold=0.01,
                              missing_threshold=0.5,
                              correlation_threshold=0.95,
                              n_features=30,
                              method='mutual_info',
                              random_state=42,
                              verbose=True):
    """
    Complete feature selection pipeline combining all three tiers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with all features
    y : pd.Series
        Target variable
    variance_threshold : float
        Minimum variance threshold (default: 0.01)
    missing_threshold : float
        Maximum missing ratio (default: 0.5)
    correlation_threshold : float
        Maximum correlation between features (default: 0.95)
    n_features : int
        Number of features to select in Tier 3 (default: 30)
    method : str
        Selection method for Tier 3 (default: 'mutual_info')
    random_state : int
        Random state for reproducibility
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    X_final : pd.DataFrame
        Final selected features
    feature_scores : pd.DataFrame
        Feature importance scores from Tier 3
    summary : dict
        Summary of the selection process
    """
    if verbose:
        print("=" * 60)
        print("FEATURE SELECTION PIPELINE")
        print("=" * 60)
        print(f"Starting with {df.shape[1]} columns\n")
    
    # Tier 1: Mandatory drops
    X, tier1_dropped = tier1_mandatory_drops(df, label_col=y.name if hasattr(y, 'name') else 'LabelBinary')
    
    # Tier 2: Statistical filtering
    X, tier2_stats = tier2_statistical_filtering(
        X, y,
        variance_threshold=variance_threshold,
        missing_threshold=missing_threshold,
        correlation_threshold=correlation_threshold
    )
    
    # Tier 3: Predictive power selection
    X_final, feature_scores = tier3_predictive_selection(
        X, y,
        method=method,
        n_features=n_features,
        random_state=random_state
    )
    
    # Create summary
    summary = {
        'original_features': df.shape[1],
        'after_tier1': len(X.columns) + sum(tier2_stats.values()),
        'after_tier2': len(X.columns),
        'final_features': len(X_final.columns),
        'tier1_dropped': len(tier1_dropped),
        'tier2_stats': tier2_stats,
        'method': method
    }
    
    if verbose:
        print(f"\n" + "=" * 60)
        print(f"FEATURE SELECTION SUMMARY")
        print("=" * 60)
        print(f"Original features:     {summary['original_features']}")
        print(f"After Tier 1:          {summary['after_tier1']}")
        print(f"After Tier 2:          {summary['after_tier2']}")
        print(f"Final selected:        {summary['final_features']}")
        print(f"Total reduction:       {summary['original_features'] - summary['final_features']} "
              f"({100*(summary['original_features'] - summary['final_features'])/summary['original_features']:.1f}%)")
        print("=" * 60)
    
    return X_final, feature_scores, summary


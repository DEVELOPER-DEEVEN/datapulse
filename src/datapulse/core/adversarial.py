"""Adversarial validation for detecting multivariate drift."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

@dataclass
class AdversarialResult:
    """Result of adversarial validation."""
    
    has_drift: bool
    auc_score: float
    threshold: float
    feature_importances: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "has_drift": self.has_drift,
            "auc_score": round(self.auc_score, 4),
            "threshold": self.threshold,
            "feature_importances": {
                k: round(v, 4) for k, v in self.feature_importances.items()
            }
        }

class AdversarialDriftDetector:
    """
    Detects multivariate drift using Adversarial Validation.
    
    The detector trains a classifier to distinguish between baseline (0) 
    and current (1) data. If the classifier can easily distinguish them 
    (high AUC), the distributions are different.
    """
    
    def __init__(self, threshold: float = 0.7, n_estimators: int = 50, cv: int = 5):
        """
        Args:
            threshold: AUC threshold to flag drift (0.5 = random/no drift, 1.0 = perfect separation)
            n_estimators: Number of trees in Random Forest
            cv: Number of cross-validation folds
        """
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.cv = cv
        
    def scan(self, baseline: pd.DataFrame, current: pd.DataFrame) -> AdversarialResult:
        """
        Run adversarial validation.
        
        Args:
            baseline: Reference data
            current: New data
        """
        # Align columns
        common_cols = list(set(baseline.columns) & set(current.columns))
        if not common_cols:
            raise ValueError("No common columns between baseline and current data")
            
        X_base = baseline[common_cols].copy()
        X_curr = current[common_cols].copy()
        
        # Labels
        y_base = np.zeros(len(X_base))
        y_curr = np.ones(len(X_curr))
        
        X = pd.concat([X_base, X_curr], axis=0).reset_index(drop=True)
        y = np.concatenate([y_base, y_curr])
        
        # Preprocessing
        # Identify categorical and numeric columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        num_cols = X.select_dtypes(include=['number']).columns
        
        transformers = []
        if len(num_cols) > 0:
            transformers.append(
                (SimpleImputer(strategy='mean'), num_cols)
            )
        if len(cat_cols) > 0:
            transformers.append(
                (make_pipeline(
                    SimpleImputer(strategy='constant', fill_value='missing'),
                    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                ), cat_cols)
            )
            
        preprocessor = make_column_transformer(*transformers, remainder='drop')
        
        # Classifier
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        pipeline = make_pipeline(preprocessor, clf)
        
        # Cross-validation AUC
        # We use StratifiedKFold to ensure balance
        cv_scores = cross_val_score(
            pipeline, X, y, 
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        mean_auc = float(np.mean(cv_scores))
        
        # Train on full set to get feature importances
        pipeline.fit(X, y)
        model = pipeline.named_steps['randomforestclassifier']
        
        # Map feature importances back to columns (approximate if encoding changes feature count, 
        # but OrdinalEncoder keeps 1-to-1)
        # Note: If OneHot was used, this would be harder. Ordinal allows direct mapping.
        
        importances = model.feature_importances_
        # The order depends on how ColumnTransformer output them.
        # Usually concatenation of transformers output.
        
        feature_names = []
        if len(num_cols) > 0:
            feature_names.extend(num_cols)
        if len(cat_cols) > 0:
            feature_names.extend(cat_cols)
            
        feat_imp_dict = {}
        if len(feature_names) == len(importances):
            feat_imp_dict = dict(zip(feature_names, importances))
            # Sort by importance
            feat_imp_dict = dict(sorted(feat_imp_dict.items(), key=lambda x: x[1], reverse=True))
        
        return AdversarialResult(
            has_drift=mean_auc > self.threshold,
            auc_score=mean_auc,
            threshold=self.threshold,
            feature_importances=feat_imp_dict
        )

#!/usr/bin/env python3
"""
Model configurations and hyperparameters for different ML algorithms.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from typing import Dict, Any, List


# Model configurations with hyperparameters and training settings
MODEL_CONFIGS = {
    'catboost': {
        'class': CatBoostClassifier,
        'params': {
            'iterations': 1000,
            'loss_function': "MultiClass",
            'grow_policy': 'Lossguide',
            'random_strength': 1.0,
            'bagging_temperature': 0.20,
            'rsm': 0.85,
            'eval_metric': "Accuracy",
            'learning_rate': 0.16,
            'random_state': 42,
            'depth': 7,
            'l2_leaf_reg': 2.5,
            'min_data_in_leaf': 12,
            'max_leaves': 20,
            'verbose': False
        },
        'use_smote': True,
        'use_class_weights': False
    },
    
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'use_smote': False,
        'use_class_weights': True
    },
    
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        },
        'use_smote': False,
        'use_class_weights': True
    },
    
    'naive_bayes': {
        'class': GaussianNB,
        'params': {
            'var_smoothing': 1e-9
        },
        'use_smote': False,
        'use_class_weights': True
    },
    
    'stacking': {
        'class': StackingClassifier,
        'params': {
            'estimators': [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ('nb', GaussianNB())
            ],
            'final_estimator': LogisticRegression(random_state=42),
            'cv': 5,
            'n_jobs': -1
        },
        'use_smote': False,
        'use_class_weights': True
    }
}

# Feature groups for different feature sets
FEATURE_GROUPS = {
    'odds_form_teams': {
        'categorical': ['HomeTeam', 'AwayTeam'],
        'form': [
            'HomeTeam_points', 'AwayTeam_points',
            'HomeTeam_avg_goal_diff', 'AwayTeam_avg_goal_diff'
        ],
        'bookmaker_odds': [
            'BWH', 'BWD', 'BWA',
            'IWH', 'IWD', 'IWA',
            'WHH', 'WHD', 'WHA',
            'VCH', 'VCD', 'VCA',
            'PSCH', 'PSCD', 'PSCA'
        ]
    },
    
    'full_features': {
        'categorical': ['HomeTeam', 'AwayTeam'],
        'form': [
            'HomeTeam_points', 'AwayTeam_points',
            'HomeTeam_avg_goal_diff', 'AwayTeam_avg_goal_diff'
        ],
        'bookmaker_odds': [
            'BWH', 'BWD', 'BWA',
            'IWH', 'IWD', 'IWA',
            'WHH', 'WHD', 'WHA',
            'VCH', 'VCD', 'VCA',
            'PSCH', 'PSCD', 'PSCA'
        ],
        'overrounds': [
            'B365_overround', 'BW_overround', 'IW_overround', 'WH_overround',
            'VC_overround', 'Max_overround', 'PS_overround', 'PSC_overround'
        ],
        'elo': ['home_elo', 'away_elo', 'elo_diff'],
        'consensus': ['pH_mean', 'pD_mean', 'pA_mean', 'overround_mean', 'overround_std'],
        'engineered': ['home_adv', 'draw_tightness']
    }
}

def get_feature_list(feature_set: str) -> List[str]:
    """
    Get list of features for a given feature set.
    
    Args:
        feature_set: Name of the feature set
        
    Returns:
        List of feature names
    """
    if feature_set not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    
    features = []
    for group_name, group_features in FEATURE_GROUPS[feature_set].items():
        features.extend(group_features)
    
    return features

def get_categorical_features(feature_set: str) -> List[str]:
    """
    Get categorical features for a given feature set.
    
    Args:
        feature_set: Name of the feature set
        
    Returns:
        List of categorical feature names
    """
    if feature_set not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    
    return FEATURE_GROUPS[feature_set].get('categorical', [])

from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier

# 3. Evaluate

model_config = {
    "xgb_baseline": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss",
        "enable_categorical": True,
    },
    "xgb_weighted": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss",
        "enable_categorical": True,
    },
     "xgb_big": {
        "n_estimators": 2000,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss",
        "enable_categorical": True,
    },
}

def train_xgboost(X_train, y_train, config="xgb_baseline"):
    """Trains an XGBoost model."""
    
    params = model_config[config]
    model = XGBClassifier(**params)

    
    # Train model
    if config == "xgb_weighted":
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight("balanced", y_train)

        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    return model


def train_xgboost_params(
    X_train,
    y_train,
    params,
    random_state=42,
    sample_weight=None,
    eval_set=None,
    early_stopping_rounds=None,
):
    """Train XGBoost with explicit params and optional weights."""
    model = XGBClassifier(
        **params,
        random_state=random_state,
        eval_metric="mlogloss",
        enable_categorical=True,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    return model


def train_ebm(X_train, y_train):
    """Trains an explainable boosting model"""
    # 1. Initialize the EBM
    # interactions=10 allows the model to find 10 pairs of features that work together
    ebm = ExplainableBoostingClassifier(
        feature_names=X_train.columns.tolist(),
        interactions=2, 
        random_state=42
    )
    print("Training EBM... This may take a few minutes.")
    
    ebm.fit(X_train, y_train)

    return ebm
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
        "enable_categorical": True
    },
}
model_config["xgb_weighted"] = model_config["xgb_baseline"].copy()

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


def train_ebm(X_train, y_train):
    """Trains an explainable boosting model"""
    # 1. Initialize the EBM
    # interactions=10 allows the model to find 10 pairs of features that work together
    ebm = ExplainableBoostingClassifier(
        feature_names=X_train.columns.tolist(),
        interactions=10, 
        random_state=42
    )
    
    ebm.fit(X_train, y_train)

    return ebm
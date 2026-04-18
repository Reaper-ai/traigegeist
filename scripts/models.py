import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import lightgbm as lgb
from imblearn.ensemble import BalancedRandomForestClassifier
import mord
from sklearn.linear_model import SGDClassifier, LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV



warnings.filterwarnings("ignore")

# ---------- Helper functions for modelling and evaluation ----------

def qwk_score(y_true, y_pred):
    """Quadratic Weighted Kappa for ordinal triage evaluation."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def compute_sample_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, weights))
    return np.array([weight_map[val] for val in y_train])

def fit_cutpoints(y_true, y_pred, n_classes):
    df_cut = pd.DataFrame({"y": y_true, "pred": y_pred})
    medians = df_cut.groupby("y")["pred"].median().sort_index()
    if len(medians) < n_classes:
        return np.quantile(y_pred, np.linspace(0, 1, n_classes + 1)[1:-1])
    cutpoints = []
    for i in range(n_classes - 1):
        cutpoints.append((medians.iloc[i] + medians.iloc[i + 1]) / 2.0)
    return np.array(cutpoints)

def apply_cutpoints(y_pred, cutpoints):
    return np.digitize(y_pred, cutpoints, right=False)

def category_codes(df_in):
    df_out = df_in.copy()
    cat_cols = df_out.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        df_out[col] = df_out[col].cat.codes
    return df_out.fillna(-1)


def eval_regressor_with_cutpoints(model, X, y, name, n_classes=2,year_bucket=None, splitter=None):
    oof_class = np.zeros(len(y), dtype=int)
    groups = year_bucket.loc[X.index]
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        sample_weight = compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        cutpoints = fit_cutpoints(y_train, train_pred, n_classes)
        val_class = apply_cutpoints(val_pred, cutpoints)
        oof_class[val_idx] = val_class
        fold_qwk = qwk_score(y_val, val_class)
        print(f"{name} fold {fold} qwk: {fold_qwk:.4f}")
    oof_qwk = qwk_score(y, oof_class)
    print(f"{name} OOF qwk: {oof_qwk:.4f}")
    return oof_class

def eval_balanced_rf(model, X, y, name):
    oof_class = np.zeros(len(y), dtype=int)
    class_values = np.arange(n_classes)
    groups = year_bucket.loc[X.index]
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        sample_weight = compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        train_proba = model.predict_proba(X_train)
        val_proba = model.predict_proba(X_val)
        train_pred = train_proba @ class_values
        val_pred = val_proba @ class_values
        cutpoints = fit_cutpoints(y_train, train_pred, n_classes)
        val_class = apply_cutpoints(val_pred, cutpoints)
        oof_class[val_idx] = val_class
        fold_qwk = qwk_score(y_val, val_class)
        print(f"{name} fold {fold} qwk: {fold_qwk:.4f}")
    oof_qwk = qwk_score(y, oof_class)
    print(f"{name} OOF qwk: {oof_qwk:.4f}")
    return oof_class


# ---------------- Model definitions ----------------

# ordinal regression models with cutpoint optimization
def xgb_reg(seed=42):
    return xgb.XGBRegressor( n_estimators=600,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_alpha=0.0,
                            reg_lambda=1.0,
                            random_state=seed,
                            n_jobs=-1,
                            tree_method="hist",
                            enable_categorical=True,
                           )

def lgb_reg(seed=42):
    return lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )

# classifer models, only for albation studies, not final evaluation
def xgb_clf(seed=42, class_weights=None):
    return xgb.XGBClassifier( n_estimators=600,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_alpha=0.0,
                            reg_lambda=1.0,
                            random_state=seed,
                            n_jobs=-1,
                            class_weight=class_weights,
                            eval_metric="mlogloss",
                            tree_method="hist",
                            enable_categorical=True,
                           )
def lgb_clf(seed=42, is_unbalanced=False):
    return lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        is_unbalanced=is_unbalanced,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
    )

def balanced_rf_clf(seed=42):
    return BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )


# keyword tagging models 
def mord_ordinal_clf(seed=42):
    return mord.LogisticAT(alpha=1.0), mord.LogisticIT(alpha=1.0)

def sgd_ordinal_clf(seed=42):
    return SGDClassifier(loss="log_loss",
                        penalty="l2",
                        alpha=1e-4,
                        class_weight="balanced",
                        max_iter=1000,
                        n_jobs=-1,
                        random_state=seed,
                        )

def linear_svc_clf(seed=42, c=5):
   linear_svc = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=1000)
   return CalibratedClassifierCV(estimator=linear_svc,method="sigmoid",cv=c, n_jobs=-1)


# meta leaner models
def meta_logistic_clf(seed=42):
    return LogisticRegression(
        solver="saga",
        max_iter=500,
        C=0.25,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )

def meta_sgd_clf(seed=42):
    return SGDClassifier(loss="log_loss",
                        penalty="l2",
                        alpha=1e-4,
                        class_weight="balanced",
                        max_iter=1000,
                        n_jobs=-1,
                        random_state=seed,
                        ) 

def stacking_regressor(seed=42):
    pass



import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import lightgbm as lgb
import lightgbm as lgb
import mord
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin



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
def meta_logistic_clf(seed=42, C=0.25, penalty='l2', l1_ratio=None):
    """The optimized meta-learner found in the notebook."""
    return LogisticRegression(
        solver="saga",
        max_iter=800,
        C=C,
        penalty=penalty,
        l1_ratio=l1_ratio,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )

class SimpleWeightedAverager(BaseEstimator, ClassifierMixin):
    """
    A simple meta-learner that averages base model probabilities.
    Weights can be uniform (Simple Average) or customized.
    """
    def __init__(self, weights=None, classes=None):
        self.weights = weights
        self.classes_ = classes

    def fit(self, X, y):
        self.classes_ = len(np.unique(y))
        return self

    def predict_proba(self, X):
        # Assume X contains stacked probabilities from N models
        n_models = X.shape[1] // self.classes_
        # Reshape to (samples, n_models, n_classes)
        probs = X.reshape(X.shape[0], n_models, self.classes_)
        
        if self.weights is None:
            return np.mean(probs, axis=1)
        
        weights = np.array(self.weights).reshape(1, n_models, 1)
        weighted_probs = np.sum(probs * weights, axis=1) / np.sum(self.weights)
        return weighted_probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

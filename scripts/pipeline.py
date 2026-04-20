import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

import data_processing as dp
import models as mod
from helpers import save_model_results

# Setup
PROJECT_ROOT = Path(__file__).parent.parent.resolve()  # Navigate from scripts/ to project root
RANDOM_STATE = 42




def run_pipeline():
    # 1. LOAD & PROCESS
    df = pd.read_csv(PROJECT_ROOT / "working_data" / "nhamcs_data_2018_22.csv")
    df_nlp = pd.read_csv(PROJECT_ROOT / "working_data" / "nlp_oof_logits_probs.csv")
    df_keyword_flags = pd.read_csv(PROJECT_ROOT / "working_data" / "nhamcs_emergency_keyword_flags_matched_only.csv")
    
    df = dp.apply_cyclical_encoding(df)
    df = dp.apply_clinical_ratios(df)
    df = dp.convert_categorical_to_numeric(df)
    dp.drop_leaky_features(df)
    dp.exclude_bias_features(df)

    df.drop(columns=["chief_complaint_text", "injury_cause_text"], inplace=True)  
    
    
    y = df["target_triage_acuity"].map(dp.map_esi).dropna().astype(int)
    X = df.loc[y.index].drop(columns=["target_triage_acuity"], errors='ignore')
    X_nlp = df_nlp.loc[y.index].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    # 2. CROSS-VALIDATION SETUP
    year_groups = df.loc[y.index, "year"]
    years_sorted = np.sort(year_groups.unique())
    year_bins = np.array_split(years_sorted, 3)
    year_bucket_map = {yr: i for i, yrs in enumerate(year_bins) for yr in yrs}
    groups = year_groups.map(year_bucket_map)
    splitter = GroupKFold(n_splits=3)
    
    n_classes = 3
    base_model_probs = {name: np.zeros((len(y), n_classes)) for name in ["xgb", "lgb", "brf"]}
    
    # 3. GENERATE OOF PROBABILITIES
    print("Generating Base Model OOF Predictions...")
    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        sw = mod.compute_sample_weights(y_train)
        
        # XGB
        m_xgb = mod.xgb_reg(RANDOM_STATE)
        m_xgb.fit(X_train, y_train, sample_weight=sw)
        # Simplified probability conversion for regressor
        base_model_probs["xgb"][val_idx, :] = np.eye(n_classes)[np.clip(m_xgb.predict(X_val).round(), 0, 2).astype(int)]
        
        # LGB
        m_lgb = mod.lgb_reg(RANDOM_STATE)
        m_lgb.fit(X_train, y_train, sample_weight=sw)
        base_model_probs["lgb"][val_idx, :] = np.eye(n_classes)[np.clip(m_lgb.predict(X_val).round(), 0, 2).astype(int)]

    # 4. STACKING / META-LEARNING
    meta_X = np.hstack([base_model_probs["xgb"], base_model_probs["lgb"], X_nlp.to_numpy()])
    
    # Variant A: Logistic Regression (The "Winner")
    print("\nFitting Final Logistic Meta-Learner...")
    meta_lr = mod.meta_logistic_clf(RANDOM_STATE)
    sw_meta = mod.compute_sample_weights(y)
    meta_lr.fit(meta_X, y, sample_weight=sw_meta)
    
    import joblib
    models_dir = PROJECT_ROOT / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_lr, models_dir / "meta_learner_final.joblib")
    print(f"Saved final meta-learner to {models_dir / 'meta_learner_final.joblib'}")

    # Variant B: Simple Average
    print("Evaluating Simple Average Variant...")
    avg_model = mod.SimpleWeightedAverager()
    # Average only the model columns of meta_X (first 6 columns for 2 models * 3 classes)
    avg_probs = avg_model.predict_proba(meta_X[:, :6])
    avg_preds = np.argmax(avg_probs, axis=1)
    
    # 5. SAVE
    save_model_results(y, avg_preds, "Simple_Average_Stack", X.columns)
    print("Pipeline Complete.")

if __name__ == "__main__":
    run_pipeline()
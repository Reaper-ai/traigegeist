import joblib
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import warnings
import os
import io
from transformers import AutoTokenizer

# Suppress transformers __path__ deprecation warnings
warnings.filterwarnings('ignore', message=".*Accessing `__path__` from.*")
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set environment variables to suppress HuggingFace warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress HuggingFace deprecation messages printed to stderr
class WarningFilter:
    def __init__(self, stream):
        self.stream = stream
    
    def write(self, msg):
        if 'Accessing `__path__` from' not in msg and 'image_processing' not in msg:
            self.stream.write(msg)
        return len(msg)
    
    def flush(self):
        self.stream.flush()
    
    def isatty(self):
        return self.stream.isatty() if hasattr(self.stream, 'isatty') else False

# Apply warning filter to stderr
sys.stderr = WarningFilter(sys.stderr)

# Add current directory to path to import scripts
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nlp import TriageModel, expand_text
from scripts.keyword_extractor import EmergencyTextProcessor, TARGET_KEYWORDS

RANDOM_STATE = 42

def map_esi(val):
    if val in (1, 2): return 0
    if val == 3: return 1
    if val in (4, 5): return 2
    return np.nan

class TriageInference:
    def __init__(self):
        self.models_dir = PROJECT_ROOT / "results" / "models"
        self.nlp_artifact_dir = PROJECT_ROOT / "results" / "model_artifacts" / "nlpie-distil-clinicalbert_corn_seed42"
        
        self.tokenizer = None
        self.nlp_model = None
        self.xgb_reg = None
        self.lgb_reg = None
        self.xgb_cls = None
        self.lgb_cls = None
        self.meta_learner = None
        
        self.centers = {}
        self.tau = {}
        
        self.processor = EmergencyTextProcessor(TARGET_KEYWORDS)

    def load_models(self):
        # Load NLP
        self.tokenizer = AutoTokenizer.from_pretrained(self.nlp_artifact_dir / "tokenizer")
        self.nlp_model = TriageModel("nlpie/distil-clinicalbert", num_classes=3)
        self.nlp_model.load_state_dict(torch.load(self.nlp_artifact_dir / "model_state.pt", map_location="cpu"))
        self.nlp_model.eval()

        # Load Tabular
        self.xgb_reg = joblib.load(self.models_dir / "xgb_reg_final.joblib")
        self.lgb_reg = joblib.load(self.models_dir / "lgb_reg_final.joblib")
        self.xgb_cls = joblib.load(self.models_dir / "xgb_cls_final.joblib")
        self.lgb_cls = joblib.load(self.models_dir / "lgb_cls_final.joblib")
        self.meta_learner = joblib.load(self.models_dir / "stacked_meta_final.joblib")

        # Calculate or load soft proba centers/tau
        # Optimized values from comprehensive_triage_modelling.ipynb logic
        self.centers['xgb'] = np.array([0.432, 1.005, 1.512])  
        self.tau['xgb'] = 0.165
        self.centers['xgb_reg'] = self.centers['xgb']
        self.tau['xgb_reg'] = self.tau['xgb']

        self.centers['lgb'] = np.array([0.445, 1.012, 1.488])
        self.tau['lgb'] = 0.170
        self.centers['lgb_reg'] = self.centers['lgb']
        self.tau['lgb_reg'] = self.tau['lgb']

    def _ensure_models_loaded(self):
        if any(
            model is None for model in (
                self.tokenizer,
                self.nlp_model,
                self.xgb_reg,
                self.lgb_reg,
                self.xgb_cls,
                self.lgb_cls,
                self.meta_learner,
            )
        ):
            raise RuntimeError("Models are not loaded. Call load_models() before predict().")

    def regressor_soft_proba(self, val_pred, name):
        key_map = {
            "xgb_reg": "xgb",
            "lgb_reg": "lgb",
            "xgb": "xgb",
            "lgb": "lgb",
        }
        key = key_map.get(name, name)
        if key not in self.centers or key not in self.tau:
            raise RuntimeError(f"Missing calibration parameters for regressor '{name}' (resolved key: '{key}').")
        centers = self.centers[key]
        tau = self.tau[key]
        dist = (val_pred[:, None] - centers[None, :]) ** 2
        probs = np.exp(-dist / (2.0 * tau))
        # Handle single prediction
        if probs.ndim == 2 and probs.shape[0] == 1:
            return probs / probs.sum(axis=1, keepdims=True)
        return probs / probs.sum(axis=1, keepdims=True)

    def get_nlp_probs(self, text):
        tokenizer = self.tokenizer
        nlp_model = self.nlp_model
        if tokenizer is None or nlp_model is None:
            raise RuntimeError("Models are not loaded. Call load_models() before predict().")

        clean_text = expand_text(text)
        enc = tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=100,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = nlp_model(enc["input_ids"], enc["attention_mask"])
            
            # CORN cumulative probs logic
            cond_probs = torch.sigmoid(logits)
            cum_probs = torch.cumprod(cond_probs, dim=1)
            probs = torch.zeros((1, 3))
            probs[0, 0] = 1.0 - cum_probs[0, 0]
            probs[0, 1] = cum_probs[0, 0] - cum_probs[0, 1]
            probs[0, 2] = cum_probs[0, 1]
            return torch.clamp(probs, min=0.0).numpy()

    def preprocess_input(self, input_data):
        self._ensure_models_loaded()
        xgb_reg = self.xgb_reg
        assert xgb_reg is not None

        df = pd.DataFrame([input_data])
        
        # 1. Basic preprocessing (Time and Clinical Ratios)
        if 'arrival_time' in df.columns:
            arrival_time = df['arrival_time'].iloc[0]
            hours = arrival_time // 100
            minutes = arrival_time % 100
            df['arrival_hour_float'] = hours + (minutes / 60.0)
            df['arrival_hour'] = hours
            df['is_shift_change'] = (
                ((df['arrival_hour_float'] >= 6)  & (df['arrival_hour_float'] < 8)) |
                ((df['arrival_hour_float'] >= 18) & (df['arrival_hour_float'] < 20))
            ).astype(int)

        cycle_maxes = {'visit_month': 12.0, 'day_of_week': 7.0, 'arrival_hour_float': 24.0}
        for col, max_val in cycle_maxes.items():
            if col in df.columns:
                angle = 2 * np.pi * df[col] / max_val
                df[f'{col}_sin'] = np.sin(angle)
                df[f'{col}_cos'] = np.cos(angle)

        if 'day_of_week' in df.columns:
            # app.py sends 1-7 (Mon-Sun). Saturday=6, Sunday=7.
            weekend_days = [6, 7] 
            is_weekend = df['day_of_week'].isin(weekend_days)
            df['is_weekend_night'] = (is_weekend & (df['arrival_hour_float'] >= 18)).astype(int)

        df["shock_index"] = df["heart_rate"] / df["sys_bp"].replace(0, np.nan)
        df["shock_index_age_adj"] = df["shock_index"] * np.where(df["age"] >= 65, 1.2, 1.0)
        df["map"] = (df["sys_bp"] + 2 * df["dias_bp"]) / 3
        df["pulse_pressure"] = df["sys_bp"] - df["dias_bp"]
        df["age_hr_interaction"] = df["age"] * df["heart_rate"]
        df["resp_spo2_ratio"] = df["resp_rate"] / df["spo2"].replace(0, np.nan)
        df["elderly_tachy"] = ((df["age"] >= 65) & (df["heart_rate"] > 100)).astype(int)

        history_cols = [c for c in df.columns if c.startswith("hist_")]
        df["history_count"] = df[history_cols].sum(axis=1)

        # NEWS2
        from scripts.data_processing import news2_score
        df["news2_score"] = news2_score(df)

        vital_cols = ["heart_rate", "sys_bp", "dias_bp", "resp_rate", "spo2", "temp"]
        df["vital_missing"] = df[vital_cols].isna().any(axis=1).astype(int)
        for col in vital_cols:
            df[f"{col}_missing"] = df[col].isna().astype(int)

        # 2. Keyword Flags
        flags_df = self.processor.process_dataframe(df, ["chief_complaint_text", "injury_cause_text"])
        df = pd.concat([df.reset_index(drop=True), flags_df.drop(columns=["row_index"]).reset_index(drop=True)], axis=1)

        # 3. Dimensionality & Format Alignment (Ground Truth: 73 features)
        try:
            all_features = xgb_reg.get_booster().feature_names
        except Exception as e:
            raise RuntimeError(f"Failed to get feature names from XGB model: {e}") from e
        
        X = df.reindex(columns=all_features).copy()
        
        # Validate that all features are present
        missing_features = X.columns[X.isna().all()].tolist()
        if missing_features:
            print(f"Warning: Missing features that are all NaN: {missing_features}")
        
        # Categorical specs
        cat_map = {
            'ems_arrival': ['Blank', 'No', 'Unknown', 'Yes'],
            'sex': ['Female', 'Male'],
            'episode': ['Blank', 'Follow-up visit to this ED', 'Initial visit to this ED', 'Unknown'],
            'is_injury_poison': ['Injury', 'No injury', 'Questionable', 'adverse effect of treatment', 'overdose/poisioning']
        }
        
        for col, levels in cat_map.items():
            if col in X.columns:
                try:
                    X[col] = pd.Categorical(X[col], categories=levels)
                except Exception as e:
                    print(f"Warning: Failed to convert {col} to categorical: {e}")
                    X[col] = pd.Categorical(X[col], categories=levels, remove_unused_categories=False) if X[col].dtype == 'object' else X[col]
        
        # BRF needs numeric codes
        X_codes = X.copy()
        for col in X_codes.columns:
            if isinstance(X_codes[col].dtype, pd.CategoricalDtype):
                X_codes[col] = X_codes[col].cat.codes
            elif X_codes[col].dtype in ['bool', 'object']:
                X_codes[col] = pd.to_numeric(X_codes[col], errors='coerce').fillna(0)
        
        X_codes = X_codes.fillna(-1)
        
        # XGB/LGB handle categoricals natively if in 'category' dtype
        for col in X.columns:
            if not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].fillna(-1)
                # Ensure boolean or objects are at least numeric for XGB if not cat
                if X[col].dtype == bool:
                    X[col] = X[col].astype(int)

        return X, X_codes

    def predict(self, input_data):
        self._ensure_models_loaded()
        xgb_reg = self.xgb_reg
        lgb_reg = self.lgb_reg
        xgb_cls = self.xgb_cls
        lgb_cls = self.lgb_cls
        meta_learner = self.meta_learner
        assert xgb_reg is not None
        assert lgb_reg is not None
        assert xgb_cls is not None
        assert lgb_cls is not None
        assert meta_learner is not None

        X, _ = self.preprocess_input(input_data)
        
        probs = {}
        
        # Regressors
        try:
            p_xgb = xgb_reg.predict(X)
            probs['xgb_reg'] = self.regressor_soft_proba(p_xgb, 'xgb')
        except Exception as e:
            raise RuntimeError(f"XGB regressor prediction failed: {e}") from e
        
        try:
            p_lgb = lgb_reg.predict(X)
            probs['lgb_reg'] = self.regressor_soft_proba(p_lgb, 'lgb')
        except Exception as e:
            raise RuntimeError(f"LGB regressor prediction failed: {e}") from e
        
        # Classifiers
        try:
            probs['xgb_cls'] = xgb_cls.predict_proba(X)
        except Exception as e:
            raise RuntimeError(f"XGB classifier prediction failed: {e}") from e
            
        try:
            probs['lgb_cls'] = lgb_cls.predict_proba(X)
        except Exception as e:
            raise RuntimeError(f"LGB classifier prediction failed: {e}") from e
        
        # NLP
        try:
            probs['nlp'] = self.get_nlp_probs(input_data['chief_complaint_text'])
        except Exception as e:
            raise RuntimeError(f"NLP model prediction failed: {e}") from e
        
        # Validate all required probability keys exist
        required_keys = ['xgb_cls', 'lgb_cls', 'xgb_reg', 'lgb_reg', 'nlp']
        missing_keys = [k for k in required_keys if k not in probs]
        if missing_keys:
            raise RuntimeError(f"Missing probability keys after prediction: {missing_keys}. Available keys: {list(probs.keys())}")
        
        # Stack order (No BRF):
        # xgb_cls, lgb_cls, xgb_reg, lgb_reg, nlp
        try:
            meta_X = np.hstack([
                probs['xgb_cls'],
                probs['lgb_cls'],
                probs['xgb_reg'], 
                probs['lgb_reg'], 
                probs['nlp']
            ])
        except Exception as e:
            raise RuntimeError(f"Failed to stack probabilities for meta learner: {e}") from e
        
        try:
            final_probs = meta_learner.predict_proba(meta_X)
        except Exception as e:
            raise RuntimeError(f"Meta learner prediction failed: {e}") from e
            
        final_class = np.argmax(final_probs, axis=1)[0]
        
        return {
            'final_class': final_class,
            'final_probs': final_probs[0],
            'base_probs': probs
        }

"""
Triage Acuity Classification Script
Model: DistilBERT with CORN (Conditional Ordinal Regression for Neural Networks) head
"""

import copy
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME =  "nlpie/distil-clinicalbert"
NUM_CLASSES = 3
MAX_LEN = 100
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 0.1
EPOCHS = 2
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "working_data" / "nhamcs_data_2018_22.csv"

# ==========================================
# REPRODUCIBILITY
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# DATA PREPROCESSING
# ==========================================
ABBR_MAP = {
    "sob": "shortness of breath", "cp": "chest pain", "abd": "abdominal",
    "ha": "headache", "n/v": "nausea and vomiting", "s/p": "status post",
    "w/": "with", "w/o": "without", "fx": "fracture", "lac": "laceration",
    "loc": "loss of consciousness", "mva": "motor vehicle accident",
    "uti": "urinary tract infection", "uri": "upper respiratory infection",
    "usp": "unspecified", "foo": "foot", "...": "", "oth": "other",
}

def expand_text(text: str) -> str:
    """Cleans text and expands common clinical abbreviations."""
    txt = str(text).lower().strip()
    txt = re.sub(r"\s+", " ", txt)

    # Replace slash abbreviations first
    for k in ["n/v", "s/p", "w/o", "w/"]:
        txt = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", ABBR_MAP[k], txt)

    # Replace standard abbreviations
    for abbr, full in ABBR_MAP.items():
        if "/" in abbr:
            continue
        txt = re.sub(rf"(?<![a-z0-9]){re.escape(abbr)}(?![a-z0-9])", full, txt)

    txt = re.sub(r"[^a-z0-9\s\-\.,]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def map_esi_to_group(val: int):
    """Maps 5-level ESI to 3-level target groups."""
    if val in (1, 2): return 0
    if val == 3: return 1
    if val in (4, 5): return 2
    return np.nan

def load_and_preprocess_data(data_path):
    """Loads CSV, identifies target column, cleans text, and drops NA."""
    df = pd.read_csv(data_path)
    df.columns = df.columns.astype(str).str.strip()

    TEXT_COL = "chief_complaint_text"
    TARGET_CANDIDATES = ["target_triage_acuity", "triage_acuity", "target"]
    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)

    if target_col is None:
        raise KeyError(f"Target column not found. Tried {TARGET_CANDIDATES}.")

    data = df[[TEXT_COL, target_col]].rename(columns={target_col: "target_triage_acuity"}).dropna()
    data["target_triage_acuity"] = data["target_triage_acuity"].astype(int).map(map_esi_to_group)
    data = data.dropna(subset=["target_triage_acuity"]).copy()
    data["target_triage_acuity"] = data["target_triage_acuity"].astype(int)
    data["text_clean"] = data[TEXT_COL].map(expand_text)
    
    # Keep year if available for GroupKFold (OOF generation)
    if "year" in df.columns:
        data["year"] = df["year"]
        
    return data

# ==========================================
# PYTORCH DATASETS
# ==========================================
class OrdinalTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=96):
        self.texts = texts.astype(str).tolist()
        self.labels = labels.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class TriageModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.5):
        super().__init__()
        try:
            self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load backbone: {e}")
            
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Ordinal regression requires num_classes - 1 output logits
        self.ordinal_head = nn.Linear(hidden, num_classes - 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_state = out.last_hidden_state[:, 0]
        return self.ordinal_head(self.dropout(cls_state))

# ==========================================
# CORN LOSS & EVALUATION UTILS
# ==========================================
def corn_loss(logits: torch.Tensor, y_idx: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """CORN trains each threshold conditionally: P(y > k | y >= k)."""
    total_loss = torch.tensor(0.0, device=logits.device)
    total_count = 0

    for k in range(num_classes - 1):
        mask = torch.ones_like(y_idx, dtype=torch.bool) if k == 0 else (y_idx >= k)
        if mask.sum() == 0:
            continue

        target = (y_idx[mask] > k).float()
        loss_k = F.binary_cross_entropy_with_logits(logits[mask, k], target, reduction="sum")
        total_loss += loss_k
        total_count += int(mask.sum().item())

    return total_loss / max(1, total_count)

def corn_cumulative_probs(logits: torch.Tensor) -> torch.Tensor:
    cond_probs = torch.sigmoid(logits)
    return torch.cumprod(cond_probs, dim=1)

def predict_from_logits(logits: torch.Tensor, thresholds=None, num_classes: int = 3) -> torch.Tensor:
    cum_probs = corn_cumulative_probs(logits)
    if thresholds is None:
        thresholds = torch.full((num_classes - 1,), 0.5, device=cum_probs.device, dtype=cum_probs.dtype)
    else:
        thresholds = torch.as_tensor(thresholds, device=cum_probs.device, dtype=cum_probs.dtype)
    return (cum_probs > thresholds).sum(dim=1).long()

def find_optimal_thresholds(logits: torch.Tensor, y_true, num_classes: int = 3):
    """Uses Nelder-Mead to find decision thresholds that maximize Quadratic Weighted Kappa (QWK)."""
    probs = corn_cumulative_probs(logits).detach().cpu().numpy()
    y_true = np.asarray(y_true)

    def objective(candidate):
        candidate = np.sort(np.clip(candidate, 0.01, 0.99))
        preds = (probs > candidate.reshape(1, -1)).sum(axis=1)
        return -cohen_kappa_score(y_true, preds, weights="quadratic")

    result = minimize(
        objective,
        x0=np.full(num_classes - 1, 0.5),
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4},
    )
    thresholds = np.sort(np.clip(result.x, 0.01, 0.99))
    return thresholds, result

# ==========================================
# TRAINING ENGINE
# ==========================================
def run_epoch(model, loader, optimizer=None, scheduler=None, desc="epoch", thresholds=None, return_logits=False):
    is_train = optimizer is not None
    model.train(is_train)

    all_true, all_pred, all_logits = [], [], []
    total_loss = 0.0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=desc, leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y_idx = batch["label"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = corn_loss(logits, y_idx, num_classes=NUM_CLASSES)

            if is_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            pred_idx = predict_from_logits(logits.detach(), thresholds=thresholds, num_classes=NUM_CLASSES).cpu().numpy()
            all_pred.extend(pred_idx.tolist())
            all_true.extend(y_idx.detach().cpu().tolist())
            total_loss += loss.item() * input_ids.size(0)

            if return_logits:
                all_logits.append(logits.detach().cpu())

            progress.set_postfix(loss=f"{(total_loss / max(1, len(all_true))):.4f}")

    epoch_loss = total_loss / len(loader.dataset)
    qwk = cohen_kappa_score(all_true, all_pred, weights="quadratic")
    all_logits_tensor = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, NUM_CLASSES - 1))
    
    return epoch_loss, qwk, all_true, all_pred, all_logits_tensor

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    print(f"Loading data from {DATA_PATH}")

    data = load_and_preprocess_data(DATA_PATH)
    print(f"Rows used: {len(data):,}")

    if "year" not in data.columns:
        raise KeyError("The 'year' column is missing. Temporal cross-validation requires it.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create Year Buckets for GroupKFold
    year_groups = data["year"]
    years_sorted = np.sort(year_groups.unique())
    year_bins = np.array_split(years_sorted, 3)
    year_bucket_map = {year: idx for idx, years in enumerate(year_bins) for year in years}
    year_bucket = year_groups.map(year_bucket_map)
    
    gkf = GroupKFold(n_splits=3)
    
    oof_logits = torch.zeros((len(data), NUM_CLASSES - 1), dtype=torch.float32)
    oof_preds = np.full(len(data), -1)  # Track tuned predictions for overall report
    oof_fold = np.full(len(data), -1)

    # Track best model across all folds to save later
    best_cv_qwk = -1.0
    best_cv_state = None
    best_cv_thresholds = None

    print("\n" + "="*40 + "\nStarting Temporal Cross-Validation\n" + "="*40)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(data, data["target_triage_acuity"], groups=year_bucket), start=1):
        print(f"\n--- Fold {fold} ---")
        
        X_train = data.iloc[train_idx]["text_clean"].fillna("")
        y_train = data.iloc[train_idx]["target_triage_acuity"].astype(int)
        X_val = data.iloc[val_idx]["text_clean"].fillna("")
        y_val = data.iloc[val_idx]["target_triage_acuity"].astype(int)

        train_ds = OrdinalTextDataset(X_train, y_train, tokenizer, max_len=MAX_LEN)
        val_ds = OrdinalTextDataset(X_val, y_val, tokenizer, max_len=MAX_LEN)

        # Handle Class Imbalance
        class_counts = y_train.value_counts().sort_index()
        class_count_tensor = torch.tensor([class_counts.get(i, 1) for i in range(NUM_CLASSES)], dtype=torch.float32)
        class_weights = 1.0 / torch.sqrt(class_count_tensor)
        
        sample_weights = class_weights[y_train.to_numpy(dtype=int)]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights.detach().cpu().double().tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

        # Initialize Model & Optimizer for this fold
        model_fold = TriageModel(MODEL_NAME, num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.AdamW(model_fold.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        total_steps = max(1, len(train_loader) * EPOCHS)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        # Train Fold
        for epoch in range(1, EPOCHS + 1):
            run_epoch(model_fold, train_loader, optimizer=optimizer, scheduler=scheduler, desc=f"Train F{fold} E{epoch}")

        # Validate Fold
        _, _, val_true, _, val_logits = run_epoch(
            model_fold, val_loader, desc=f"Val F{fold}", return_logits=True
        )
        
        # Calculate thresholds and QWK
        epoch_thresholds, _ = find_optimal_thresholds(val_logits, val_true, num_classes=NUM_CLASSES)
        val_pred_tuned = predict_from_logits(val_logits, thresholds=epoch_thresholds, num_classes=NUM_CLASSES).cpu().numpy()
        val_qwk = cohen_kappa_score(val_true, val_pred_tuned, weights="quadratic")
        
        # Save OOF predictions for this fold
        oof_logits[val_idx] = val_logits
        oof_preds[val_idx] = val_pred_tuned
        oof_fold[val_idx] = fold
        
        # Print Fold Results
        print(f"\nFold {fold} QWK: {val_qwk:.4f}")
        print(f"Fold {fold} Classification Report:")
        print(classification_report(val_true, val_pred_tuned, digits=4))

        # If this fold performed the best, save its state in memory
        if val_qwk > best_cv_qwk:
            best_cv_qwk = val_qwk
            best_cv_state = copy.deepcopy(model_fold.state_dict())
            best_cv_thresholds = epoch_thresholds.copy()

    # ==========================================
    # OVERALL OOF EVALUATION
    # ==========================================
    print("\n" + "="*40 + "\nOverall OOF Performance\n" + "="*40)
    
    y_true_all = data["target_triage_acuity"].astype(int).to_numpy()
    overall_qwk = cohen_kappa_score(y_true_all, oof_preds, weights="quadratic")
    
    print(f"Overall OOF QWK: {overall_qwk:.4f}")
    print("Overall OOF Classification Report:")
    print(classification_report(y_true_all, oof_preds, digits=4))

    # ==========================================
    # SAVE OOF DATAFRAME
    # ==========================================
    print("\n" + "="*40 + "\nGenerating OOF Files\n" + "="*40)
    
    # Convert logits to probabilities
    cond_probs = torch.sigmoid(oof_logits)
    cum_probs = torch.cumprod(cond_probs, dim=1)
    probs = torch.zeros((cum_probs.size(0), NUM_CLASSES), dtype=cum_probs.dtype)
    probs[:, 0] = 1.0 - cum_probs[:, 0]
    for c in range(1, NUM_CLASSES - 1):
        probs[:, c] = cum_probs[:, c - 1] - cum_probs[:, c]
    probs[:, NUM_CLASSES - 1] = cum_probs[:, NUM_CLASSES - 2]
    probs = torch.clamp(probs, min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

    # Build DataFrame
    oof_df = pd.DataFrame({
        "row_index": data.index.to_numpy(),
        "year": year_groups.to_numpy(),
        "fold": oof_fold,
        "pred_class": oof_preds
    })
    
    for k in range(NUM_CLASSES - 1):
        oof_df[f"raw_logit_t{k+1}"] = oof_logits[:, k].numpy()
    for c in range(NUM_CLASSES):
        oof_df[f"prob_class_{c+1}"] = probs[:, c].numpy()

    oof_path = PROJECT_ROOT / "working_data" / "nlp_oof_logits_probs.csv"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved {len(oof_df):,} OOF records to: {oof_path}")

    # ==========================================
    # SAVE BEST MODEL ARTIFACTS
    # ==========================================
    if best_cv_state is not None:
        print("\n" + "="*40 + "\nSaving Best Fold Model Artifacts\n" + "="*40)
        
        safe_model_name = MODEL_NAME.replace("/", "-")
        model_tag = f"{safe_model_name}_corn_seed{SEED}"
        
        artifact_dir = PROJECT_ROOT / "results" / "model_artifacts" / model_tag
        tokenizer_dir = artifact_dir / "tokenizer"
        weights_path = artifact_dir / "model_state.pt"
        metadata_path = artifact_dir / "metadata.json"

        artifact_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        torch.save(best_cv_state, weights_path)
        tokenizer.save_pretrained(tokenizer_dir)

        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_name": MODEL_NAME,
            "num_classes": int(NUM_CLASSES),
            "max_len": int(MAX_LEN),
            "dropout": 0.5,
            "seed": int(SEED),
            "weights_path": str(weights_path),
            "tokenizer_dir": str(tokenizer_dir),
            "class_labels": [0, 1, 2],
            "calibrated_thresholds": [float(x) for x in best_cv_thresholds],
            "best_cv_qwk": float(best_cv_qwk),
            "overall_oof_qwk": float(overall_qwk)
        }

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Artifacts Saved Successfully (Best Fold QWK: {best_cv_qwk:.4f}):")
        print(f"  Weights: {weights_path}")
        print(f"  Tokenizer: {tokenizer_dir}")
        print(f"  Metadata: {metadata_path}")
        
if __name__ == "__main__":
    main()
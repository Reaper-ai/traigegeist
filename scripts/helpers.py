import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKING_DATA_DIR = PROJECT_ROOT / "working_data"
FIG_DIR = PROJECT_ROOT / "fig"
RESULTS_DIR = PROJECT_ROOT / "results" / "classification_reports"

def load_embeddings(filename=None):
    """Loads embeddings if they exist, otherwise returns None."""
    if filename is None:
        filename = WORKING_DATA_DIR / "embeddings.parquet"
    filename = Path(filename)
    if filename.exists():
        print(f"🚀 Loading cached embeddings from {filename}...")
        return pd.read_parquet(filename)
    return None

def generate_embeddings(df, filename=None):
    """Generates embeddings for chief_complaint_text (30D) and injury_cause_text (20D) separately."""
    if filename is None:
        filename = WORKING_DATA_DIR / "embeddings.parquet"
    filename = Path(filename)
    cached_emb = load_embeddings(filename)
    if cached_emb is not None:
        return cached_emb
    
    print("Generating embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    df_emb = pd.DataFrame(index=df.index)
    
    # Chief complaint: embed unique, PCA to 30D
    chief_unique = df['chief_complaint_text'].dropna().unique()
    chief_emb_raw = embedder.encode(chief_unique.tolist(), show_progress_bar=True)
    pca_chief = PCA(n_components=30)
    chief_emb_pca = pca_chief.fit_transform(chief_emb_raw)
    chief_map = pd.Series(chief_emb_pca.tolist(), index=chief_unique)
    
    chief_cols = [f"chief_complaint_emb_{i}" for i in range(30)]
    df_emb[chief_cols] = df['chief_complaint_text'].map(chief_map).apply(pd.Series)
    
    # Injury cause: embed unique, PCA to 16D
    injury_unique = df['injury_cause_text'].dropna().unique()
    injury_emb_raw = embedder.encode(injury_unique.tolist(), show_progress_bar=True)
    pca_injury = PCA(n_components=16)
    injury_emb_pca = pca_injury.fit_transform(injury_emb_raw)
    injury_map = pd.Series(injury_emb_pca.tolist(), index=injury_unique)
    
    injury_cols = [f"injury_cause_emb_{i}" for i in range(16)]
    df_emb[injury_cols] = df['injury_cause_text'].map(injury_map).apply(pd.Series)
    
    # Save embeddings
    WORKING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_emb.to_parquet(filename)
    print(f"✅ Saved embeddings to {filename}")
    
    return df_emb




def save_classification_report(
    y_true,
    y_pred,
    model_name,
    seed=None,
    config=None,
    columns=None,
    notes=None,
    results_dir=None,
):
    """Save a classification report to JSON and text, keeping prior reports."""
    target_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = f"seed{seed}" if seed is not None else "seedNA"
    base = f"{model_name}_{seed_tag}_{timestamp}"
    report_text = classification_report(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    meta = {
        "model": model_name,
        "seed": seed,
        "config": config,
        "n_features": int(len(columns)) if columns is not None else None,
        "timestamp": timestamp,
        "notes": notes,
    }
    json_path = target_dir / f"{base}.json"
    txt_path = target_dir / f"{base}.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "report": report_dict}, f, indent=2, sort_keys=True)
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2, sort_keys=True))
        f.write("\n\n")
        f.write(report_text)
    print(report_text)
    return report_dict




data_split_config = {
    "base_line": {"test_size": 0.2, "columns_to_drop": ['intervention_iv_fluids', 'vitals_during_visit', 'wait_time_minutes', 'year']},
}

def data_split(df, config="base_line", random_state=42):
    """Splits the data into train and validation sets."""
    from sklearn.model_selection import train_test_split

    
    # Get config values
    test_size = data_split_config[config]["test_size"]
    cols_to_exclude = data_split_config[config]["columns_to_drop"] +  ['target_triage_acuity']

    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
            df[col] = df[col].astype('category')

    # Create target variable
    X = df.drop(columns=cols_to_exclude).copy()
    y = df['target_triage_acuity'] - 1  # Map 1-5 to 0-4

    
    # Train-test split
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return x_train, x_val, y_train, y_val

def plot_feature_importance(model, feature_names, top_n=20, output_path=None):
    """Plots the top N feature importances from the model."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not have feature importance attribute.")
        return
    
    indices = np.argsort(importances)[::-1][:top_n]
    
    if output_path is None:
        output_path = FIG_DIR / "feature_importance.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances")
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def top_k_feature_names(model, feature_names, k=50):
    """Return top-k feature names based on model feature_importances_."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:k]
    return [feature_names[i] for i in indices]


def fairness_summary_by_group(y_true, y_pred, group_series):
    """Summarize under/over-triage rates by group."""
    df_eval = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group": group_series.astype("category"),
    })
    df_eval = df_eval.dropna(subset=["group"])
    if df_eval.empty:
        return pd.DataFrame()
    df_eval["under"] = (df_eval["y_pred"] < df_eval["y_true"]).astype(int)
    df_eval["over"] = (df_eval["y_pred"] > df_eval["y_true"]).astype(int)
    df_eval["correct"] = (df_eval["y_pred"] == df_eval["y_true"]).astype(int)
    summary = df_eval.groupby("group").agg(
        n=("y_true", "size"),
        under_rate=("under", "mean"),
        over_rate=("over", "mean"),
        accuracy=("correct", "mean"),
    ).sort_values("n", ascending=False)
    return summary


def run_fairness_checks(y_true, y_pred, X_eval, group_cols):
    """Run fairness summaries for selected group columns."""
    results = {}
    for col in group_cols:
        if col in X_eval.columns:
            results[col] = fairness_summary_by_group(y_true, y_pred, X_eval[col])
    return results
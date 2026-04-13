import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def load_embeddings(filename="embeddings.parquet"):
    """Loads embeddings if they exist, otherwise returns None."""
    if os.path.exists(filename):
        print(f"🚀 Loading cached embeddings from {filename}...")
        return pd.read_parquet(filename)
    return None

def generate_embeddings(df, filename="embeddings.parquet"):
    """Generates embeddings for chief_complaint_text (30D) and injury_cause_text (20D) separately."""
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
    
    # Injury cause: embed unique, PCA to 20D
    injury_unique = df['injury_cause_text'].dropna().unique()
    injury_emb_raw = embedder.encode(injury_unique.tolist(), show_progress_bar=True)
    pca_injury = PCA(n_components=20)
    injury_emb_pca = pca_injury.fit_transform(injury_emb_raw)
    injury_map = pd.Series(injury_emb_pca.tolist(), index=injury_unique)
    
    injury_cols = [f"injury_cause_emb_{i}" for i in range(20)]
    df_emb[injury_cols] = df['injury_cause_text'].map(injury_map).apply(pd.Series)
    
    # Save embeddings
    df_emb.to_parquet(filename)
    print(f"✅ Saved embeddings to {filename}")
    
    return df_emb




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

def plot_feature_importance(model, feature_names, top_n=20):
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
    
    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances")
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
    plt.savefig("feature_importance.png")
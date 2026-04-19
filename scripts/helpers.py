import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import classification_report, cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "classification_reports"


def save_classification_report(
    y_true,
    y_pred,
    model_name,
    seed=None,
    config=None,
    columns=None,
    notes=None,
    results_dir=None,
    extra_metrics=None,
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
    if extra_metrics:
        meta.update(extra_metrics)
    json_path = target_dir / f"{base}.json"
    txt_path = target_dir / f"{base}.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "report": report_dict}, f, indent=2, sort_keys=True)
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2, sort_keys=True))
        f.write("\n\n")
        f.write(report_text)
    return report_dict


def save_model_results(y_true, y_pred, name, columns, extra_metrics=None):
    """Convenience wrapper for the project-specific naming convention."""
    from scripts.models import qwk_score
    
    metrics = extra_metrics or {}
    metrics["qwk"] = qwk_score(y_true, y_pred)
    
    return save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        model_name=name,
        columns=columns,
        extra_metrics=metrics
    )
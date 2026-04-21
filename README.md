# 🚑 Real-World Emergency Triage Risk Stratification with a Multimodal Ordinal Ensemble

3-class emergency triage decision aid built on NHAMCS 2018-2022 real-world ED visits.

This project targets a practical Kaggle-style workflow:
- 🧪 messy raw healthcare data
- 🧱 reproducible feature pipelines
- 🛡️ leakage-aware validation
- 🔎 interpretable model outputs
- 🖥️ deployable Streamlit interface

---

## ✨ TL;DR

We predict triage acuity in 3 clinically actionable classes:
- 🔴 Urgent: ESI 1-2
- 🟠 Emergent: ESI 3
- 🟢 Non-Urgent: ESI 4-5

Core idea:
- 🧠 combine tabular clinical signals, emergency keyword flags, and fine-tuned clinical NLP probabilities
- 📐 train both ordinal-aware regressors and multiclass classifiers
- 🏗️ stack everything with a logistic meta-learner

Primary paper-level summary (from project write-up):
- 🎯 weighted F1: 0.59
- 🚨 urgent recall: 0.65
- 📏 QWK: 0.5214
- ✅ accuracy: 58.82%

Notebook execution snapshot in this repo (modelling output):
- 📊 Stacked Logistic Meta-Learner OOF QWK: 0.5122
- 🎯 weighted F1: 0.5829
- 🚨 urgent recall: 0.6542

---

## 🩺 Why This Project Matters

ED triage is high-stakes and noisy in real life. Even trained staff can disagree on ESI labels, and undertriage can delay critical interventions.

This repository focuses on:
- 🌍 real public NHAMCS data (not synthetic)
- 📐 ordinal-aware modelling (misranking by 2 levels is worse than by 1)
- 🧪 robust out-of-fold evaluation
- 🧭 clinically interpretable drivers (vitals, history, complaint signals, time effects)

---

## 📊 Dataset

Source: NHAMCS emergency department public-use data (2018-2022).

Raw files:
- data/ED2018-stata.dta
- data/ED2019-stata.dta
- data/ed2020-stata.dta
- data/ed2021-stata.dta
- data/ed2022-stata.dta

Format maps:
- format/ed18for.txt
- format/ed19for.txt
- format/ed20for.txt
- format/ed21for.txt
- format/ed22for.txt

Processed cohort used across modelling notebooks:
- 58,124 rows after dropping invalid or missing triage targets

Observed ESI distribution before 3-class collapse (from notebook outputs):
- ESI 1: 846
- ESI 2: 8,597
- ESI 3: 29,568
- ESI 4: 16,715
- ESI 5: 2,398

---

## 🧩 End-to-End Data Pipeline

### 1) 📥 STATA ingestion and harmonization

Notebook: notebooks/data_processing.ipynb

What happens:
- loads all 5 STATA year files
- extracts required columns (arrival/time, demographics, vitals, RFV complaint fields, injury, chronic history, target-related columns)
- decodes coded variables using STATA label maps and format text files
- merges years into a single dataframe
- renames columns into model-ready names
- saves:
	- working_data/nhamcs_data_2018_22.csv

### 2) 🚨 Emergency keyword flag generation

Notebook: notebooks/text_processing.ipynb

What happens:
- reads working_data/nhamcs_data_2018_22.csv
- normalizes complaint/injury text
- expands abbreviations (example: sob -> shortness of breath, cp -> chest pain)
- applies strict negation handling (example: no chest pain should not trigger)
- starts from a broad emergency keyword set, then keeps matched project columns
- final selected 14 keyword flags:
	- chest_pain
	- shortness_of_breath
	- syncope
	- assault
	- vaginal_bleeding
	- violence
	- burn
	- head_injury
	- suicide_attempt
	- cardiac_arrest
	- gunshot_wound
	- throat_swelling
	- paralysis
	- sepsis
- saves:
	- working_data/nhamcs_emergency_keyword_flags_matched_only.csv

### 3) 🧠 NLP OOF probability generation

Notebook: notebooks/nlp_dl.ipynb

What happens:
- cleans chief complaint text
- fine-tunes nlpie/distil-clinicalbert with a CORN ordinal head
- performs year-bucketed GroupKFold OOF generation
- writes NLP logits/probabilities for stacking
- saves:
	- working_data/nlp_oof_logits_probs.csv
	- results/model_artifacts/distilbert_corn_seed42/*

### 4) 🏗️ Multimodal tabular + NLP stacking

Notebook: notebooks/modelling.ipynb

What happens:
- reads:
	- working_data/nhamcs_data_2018_22.csv
	- working_data/nlp_oof_logits_probs.csv
	- working_data/nhamcs_emergency_keyword_flags_matched_only.csv
- applies cyclical time features, clinical ratios, missingness flags, NEWS2 approximation
- trains base learners:
	- xgb_reg, lgb_reg (ordinal via post-hoc cutpoints)
	- xgb_cls, lgb_cls (multiclass)
	- NLP OOF probs as another base signal
- stacks with logistic regression meta-learner
- saves final model files into results/models in notebook execution

---

## 🧠 Model Architecture

### 🎯 Target setup

3-class mapping:
- 0: Urgent (ESI 1-2)
- 1: Emergent (ESI 3)
- 2: Non-Urgent (ESI 4-5)

### 🧱 Modalities

1. Tabular clinical tower
- demographics, arrival context, vitals, chronic history
- engineered features: shock index, MAP, pulse pressure, age-heart rate interaction, NEWS2 approximation
- explicit missingness indicators

2. Keyword flag tower
- emergency term indicators from complaint/injury text with negation-aware extraction

3. Chief complaint NLP tower
- Distil ClinicalBERT backbone
- CORN ordinal head
- out-of-fold class probabilities for stacking

4. Meta-learner
- multinomial logistic regression over stacked base probabilities

---

## 📈 Performance Snapshot

From notebooks/modelling.ipynb OOF output (3-class):

| Model | QWK | Weighted F1 |
|---|---:|---:|
| XGB Regressor (ordinal cutpoints) | 0.3998 | 0.4534 |
| LGBM Regressor (ordinal cutpoints) | 0.3917 | 0.4492 |
| XGB Classifier | 0.4071 | 0.4632 |
| LGBM Classifier | 0.3991 | 0.4587 |
| NLP-only signal (OOF-derived) | 0.4740 | 0.5579 |
| Simple Average Meta | 0.4552 | 0.5012 |
| Weighted Average Meta | 0.4620 | 0.5071 |
| Stacked Logistic Meta | 0.5122 | 0.5829 |

Stacked Logistic class-wise recall (OOF):
- Urgent: 0.6542
- Emergent: 0.4733
- Non-Urgent: 0.7135

---

## 🔍 Explainability Assets

### 🌍 Global/tabular interpretation

Notebook: notebooks/model_interpretation.ipynb

Includes:
- model-native feature importance plots
- SHAP summary analysis
- cross-model comparison in results/feature_importance and plot/

### 🗣️ NLP local explanations

Notebook: notebooks/nlp_lime_explanations.ipynb

Saves:
- per-case HTML LIME reports in results/lime
- keyword weight CSVs in results/lime
- visualization PNGs in plot/

---

## 🖥️ Streamlit App

File: app.py

UI summary:
- chief complaint text box
- vitals input panel
- history/comorbidity toggles
- context inputs (arrival date/time, EMS, seen in last 72h)
- triage card output with confidence and per-class probability bars
- expandable base-model probability table

Inference backend:
- utils.py -> TriageInference class
- combines xgb/lgb regressors + classifiers + NLP probabilities
- final prediction via stacked meta-learner

---

## 🗂️ Project Layout

```text
traigegeist/
├── app.py
├── utils.py
├── pyproject.toml
├── data/                      # raw NHAMCS STATA files (2018-2022)
├── format/                    # STATA format maps
├── working_data/              # processed CSVs used by modelling
├── notebooks/
│   ├── data_processing.ipynb
│   ├── text_processing.ipynb
│   ├── eda.ipynb
│   ├── nlp_dl.ipynb
│   ├── modelling.ipynb
│   ├── comprehensive_triage_modelling.ipynb
│   ├── model_interpretation.ipynb
│   └── nlp_lime_explanations.ipynb
├── scripts/                   # reusable pipeline modules
├── results/
│   ├── classification_reports/
│   ├── feature_importance/
│   ├── lime/
│   └── model_artifacts/
└── plot/                      # generated EDA + interpretation figures
```

---

## ▶️ Notebook Run Order (Recommended)

1. notebooks/data_processing.ipynb
2. notebooks/text_processing.ipynb
3. notebooks/nlp_dl.ipynb
4. notebooks/modelling.ipynb
5. notebooks/model_interpretation.ipynb
6. notebooks/nlp_lime_explanations.ipynb
7. notebooks/eda.ipynb (exploration/reporting at any point after step 1)

---

## ⚙️ Local Setup

### 📦 Requirements

- Python 3.12+
- dependencies from pyproject.toml

### 🛠️ Install

Using pip:

```bash
pip install -e .
```

### 🚀 Run Streamlit app

```bash
streamlit run app.py
```

---

## ♻️ Notes for Reproducibility

- The modelling notebook writes final tabular/meta models to results/models.
- The NLP notebook writes model artifacts to results/model_artifacts/distilbert_corn_seed42.
- The current utils.py app loader expects an NLP artifact folder named:
	- results/model_artifacts/nlpie-distil-clinicalbert_corn_seed42

If you only have distilbert_corn_seed42, either:
- rerun scripts/nlp.py (which uses the safe model-name tag), or
- update the artifact path in utils.py.

---

## 🏁 Kaggle Workflow Notes

This repository is organized for competition-style iteration:

1. Build stable feature assets once
- run data processing + text processing notebooks
- keep working_data CSVs versioned or checkpointed

2. Iterate on towers independently
- tabular experiments in notebooks/modelling.ipynb
- NLP experiments in notebooks/nlp_dl.ipynb

3. Stack and compare
- track QWK, weighted F1, and urgent recall together
- keep every report in results/classification_reports

4. Explain before submit
- verify key drivers in notebooks/model_interpretation.ipynb
- review local text behavior in notebooks/nlp_lime_explanations.ipynb

5. Demo package
- run Streamlit app for qualitative case checks before finalizing model snapshots

---

## 📚 Citation

If you use this repository, cite NHAMCS and the methods used in this project write-up (ordinal modelling, stacked ensembling, SHAP, and LIME).

Key methods and references are documented in the project manuscript draft and notebook outputs.


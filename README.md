# Calibrated Churn Prediction with Cost-Sensitive Evaluation

**Intermediate title:** Calibrated Churn Prediction with Cost-Sensitive Evaluation  
**What this project demonstrates:** data generation, leakage checks, feature engineering, model training, probability calibration (Platt/Isotonic), threshold optimization for business value, SHAP-based explanations, simple fairness slices, batch inference, a lightweight FastAPI inference service, and a Streamlit "uplift simulator".

## Structure
```
calibrated_churn_project/
├─ data/                       # synthetic sample dataset (csv)
├─ notebooks/                  # short EDA & quick train notebook (not executable here)
├─ src/
│  ├─ data.py                  # data generation & leakage checks
│  ├─ features.py              # feature engineering helpers
│  ├─ model.py                 # training, calibration, saving model
│  ├─ fairness.py              # simple fairness metrics
│  └─ utils.py                 # utility functions
├─ app/
│  ├─ api.py                   # FastAPI inference service
│  └─ requirements.txt
├─ streamlit_app/
│  └─ app.py                   # simple uplift simulator UI
├─ train.sh                    # example train commands
├─ requirements.txt
└─ README.md
```

## Quickstart (local)
1. Create virtualenv and install:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2. Generate synthetic data:
```bash
python -m src.data --output data/sample_churn.csv --n 5000
```
3. Train a model (this will create `artifacts/model.joblib` and `artifacts/calibrator.joblib`):
```bash
python -m src.model --input data/sample_churn.csv --output_dir artifacts
```
4. Run the FastAPI service:
```bash
cd app
uvicorn api:app --reload --port 8000
```
5. Run Streamlit uplift simulator:
```bash
streamlit run streamlit_app/app.py
```

## Notes
- This is intentionally **not** perfect: some hyperparameters are simple, the dataset is synthetic, and production-level concerns (auth, monitoring) are omitted.
- The repository includes instructions and scripts to reproduce training and run lightweight demos.

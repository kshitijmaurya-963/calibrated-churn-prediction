## Calibrated Churn Prediction

An **end-to-end data science project** for customer churn prediction that goes beyond standard classification.  
This project demonstrates how to handle **probability calibration, business-driven decision thresholds, and explainability**, all packaged in a reproducible and modular structure.

---

**🚀 Project Highlights**

- **Data Preprocessing & Leakage Checks**
  - Robust handling of categorical & numerical features
  - Ensures reproducibility with artifact saving

- **Model Training & Calibration**
  - Trains baseline classifiers (Logistic Regression / Tree-based)
  - Applies **Platt scaling (sigmoid)** or **Isotonic regression** for probability calibration
  - Optimizes decision thresholds for **expected business value** (not just accuracy)

- **Evaluation Metrics**
  - ROC-AUC, Brier score, calibration curve
  - Cost-sensitive evaluation of churn vs retention

- **Explainability**
  - SHAP-based feature importance
  - Local explanations for individual customers

- **Interactive Dashboard**
  - Built with **Streamlit**
  - Upload new customer data and obtain churn probability + calibrated decision
  - Visualize feature contributions for transparency

---

**🗂️ Project Structure**
```
calibrated-churn-prediction/
│── data/ # Sample input data
│ └── sample_churn.csv
│── artifacts/ # Trained models, encoders, calibration outputs
│── notebooks/ # EDA and experimentation
│── src/ # Core source code
│ ├── init.py
│ ├── features.py # Feature engineering
│ ├── model.py # Training, calibration, evaluation
│ ├── utils.py # Helpers (metrics, plotting, etc.)
│── streamlit_app/ # Dashboard
│ └── app.py
│── requirements.txt
│── README.md
```

---

**⚙️ Installation & Setup**

### 1️. Clone the Repository
```bash
git clone https://github.com/<your-username>/calibrated-churn-prediction.git
cd calibrated-churn-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Model & Generate Artifacts
```bash
python -m src.model --input data/sample_churn.csv --output_dir artifacts
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run streamlit_app/app.py
```

---

**📈 What This Project Demonstrates**   
✔️ Handling imbalanced churn data with probability calibration   
✔️ Making models actionable for business decisions via threshold optimization  
✔️ Interpretable ML using SHAP values  
✔️ Modular design suitable for MLOps / productionization  
✔️ Interactive UI to engage stakeholders beyond Jupyter notebooks  

**🔮 Future Scope**    
- Add support for multiple base classifiers (XGBoost, LightGBM, CatBoost)  
- Extend dashboard with customer cohort analysis  
- Deploy containerized app with Docker + CI/CD  
- Integrate with real-world churn datasets (telecom, SaaS, banking)  

**🤝 Contributing**  
Contributions are welcome! Please fork the repo and submit a PR with improvements.

---

**👨‍💻 Author**    
Kshitij Maurya    
Data Scientist | AI/ML Engineer | Product Thinker  

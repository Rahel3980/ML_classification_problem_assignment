# Employee Attrition Prediction — The ML Process

Predicts which employees are likely to leave a company, so HR can act early.
Supervised binary classification on the **IBM HR Analytics Employee Attrition** dataset (Kaggle, 1,470 employees).

**Final model:** tuned Logistic Regression (C = 0.03, L2), decision threshold 0.54.

| Test-set metric | Target | Result |
|---|---|---|
| ROC-AUC | ≥ 0.80 | **0.825** |
| Recall (leavers) | ≥ 0.70 | **0.722** |
| F1 (leavers) | ≥ 0.50 | **0.510** |

## Project structure

```
ML_Classification_Problem/
├── notebooks/
│   ├── 01_eda.ipynb          # 1 Problem, 2 Data, 3 EDA + cleaning  -> saves data_ibm/attrition_clean.csv
│   └── 02_model.ipynb        # 3 Features/split/scaling, 4-6 Models + tuning, 8 Test + save, 10 Monitoring
├── app/app.py                # Streamlit app: single prediction, batch CSV scoring, monitoring dashboard
├── data_ibm/                 # raw IBM CSV + cleaned CSV
├── attrition_deployment_bundle.pkl   # saved preprocessor + model + threshold (created by 02_model)
├── reports/
│   ├── figures/              # all charts (notebooks) + slide versions and app screenshot
│   ├── Process_Documentation.docx             # written report of every step
│   └── Employee_Attrition_Presentation.pptx   # 14-slide presentation
├── monitoring/               # prediction and feedback logs (created by the app at runtime)
├── archive/                  # first attempt on a synthetic dataset with no signal (kept for reference)
└── requirements.txt
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 1. Notebooks (run in this order)
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_model.ipynb

# 2. App
streamlit run app/app.py
```

No paths need editing: the notebooks find `data_ibm/` whether started from the project folder or from `notebooks/`.

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a GitHub repository (the `.pkl` file and `requirements.txt` must be included).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose the repository, branch `main`, main file path `app/app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` and gives the app a public URL.

Note: on Streamlit Cloud the monitoring log files are reset whenever the app restarts. For permanent
monitoring the logs would go to a database; for this assignment the in-app dashboard and CSV download are enough.

## Monitoring

The app's **Monitoring** tab shows: number of predictions, response latency (median and 95th percentile),
share flagged high-risk compared with training, a data-drift table (shift of each input's mean from the
training mean in standard deviations, flagged above 0.5) and HR feedback (real outcomes → real-world accuracy and recall).

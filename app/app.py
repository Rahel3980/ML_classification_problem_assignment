"""Employee Attrition Early-Warning System (Streamlit app).

Run from the project folder:  streamlit run app/app.py
"""
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent          # project folder
BUNDLE_PATH = ROOT / "attrition_deployment_bundle.pkl"
LOG_PATH = ROOT / "monitoring" / "prediction_log.csv"
FEEDBACK_PATH = ROOT / "monitoring" / "feedback_log.csv"
DRIFT_LIMIT = 0.5   # flag a feature when its mean moves more than 0.5 training SDs


# ----------------------------------------------------------
# FEATURE ENGINEERING - copied from notebooks/02_model.ipynb (keep identical)
# ----------------------------------------------------------
def add_features(df):
    out = df.copy()
    overtime = (out["OverTime"] == "Yes").astype(int)
    out["Overall_Satisfaction"] = out[["EnvironmentSatisfaction", "JobSatisfaction",
                                       "RelationshipSatisfaction", "WorkLifeBalance"]].mean(axis=1)
    out["Income_per_Level"] = out["MonthlyIncome"] / out["JobLevel"]
    out["Tenure_Ratio"] = (out["YearsAtCompany"] / out["TotalWorkingYears"].replace(0, np.nan)).fillna(0).clip(0, 1)
    out["Promotion_Stagnation"] = out["YearsSinceLastPromotion"] / (out["YearsAtCompany"] + 1)
    out["Role_Stagnation"] = out["YearsInCurrentRole"] / (out["YearsAtCompany"] + 1)
    out["Job_Hopping_Rate"] = out["NumCompaniesWorked"] / (out["TotalWorkingYears"] + 1)
    out["Overtime_Strain"] = overtime * (5 - out["WorkLifeBalance"])
    out["Age_Group"] = pd.cut(out["Age"], bins=[0, 29, 39, 49, 200],
                              labels=["Under 30", "30-39", "40-49", "50+"]).astype(str)
    return out


# ----------------------------------------------------------
# PAGE SETUP & MODEL LOADING
# ----------------------------------------------------------
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="📊", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Could not load the model file `{BUNDLE_PATH.name}` from the project folder. "
             f"Run notebooks/02_model.ipynb first. Error: {e}")
    st.stop()

preprocessor, model = bundle["preprocessor"], bundle["model"]
threshold = float(bundle["threshold"])
ranges, categories = bundle["numeric_ranges"], bundle["categories"]
RAW_FEATURES = bundle["raw_features"]


def predict(raw_df):
    """Return attrition probabilities for a DataFrame of raw employee records."""
    X = preprocessor.transform(add_features(raw_df[RAW_FEATURES]))
    return model.predict_proba(X)[:, 1]


def drift_table(new_df):
    stats = bundle["train_stats"]
    cols = list(stats["mean"])
    train_mean, train_std = pd.Series(stats["mean"]), pd.Series(stats["std"])
    new_mean = new_df[cols].astype(float).mean()
    out = pd.DataFrame({"Training mean": train_mean, "New mean": new_mean,
                        "Shift (SD)": (new_mean - train_mean) / train_std}).round(2)
    out["Status"] = np.where(out["Shift (SD)"].abs() > DRIFT_LIMIT, "⚠️ Drift", "OK")
    return out.reindex(out["Shift (SD)"].abs().sort_values(ascending=False).index)


def append_csv(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(path, mode="a", header=not path.exists(), index=False)


# ----------------------------------------------------------
# HEADER & SIDEBAR (all numbers come from the saved model file)
# ----------------------------------------------------------
st.title("📊 Employee Attrition Early-Warning System")
st.caption(f"Predicts the probability that an employee will leave · {bundle['model_name']} · "
           f"flag threshold {threshold:.0%}")

m = bundle["test_metrics"]
with st.sidebar:
    st.header("⚙️ Model")
    st.markdown(f"**Algorithm:** {bundle['model_name']}  \n"
                f"**Parameters:** `{bundle['best_params']}`  \n"
                f"**Decision threshold:** {threshold:.0%}")
    st.subheader("Test-set performance")
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    c2.metric("Recall", f"{m['Recall']:.0%}")
    c1.metric("Precision", f"{m['Precision']:.0%}")
    c2.metric("F1", f"{m['F1']:.3f}")
    st.caption("Trained on the IBM HR Analytics dataset (1,470 employees). "
               "Use the score to prioritise conversations, not to make decisions about people.")

tab_single, tab_batch, tab_monitor = st.tabs(
    ["👤 Single employee", "📁 Batch scoring (CSV)", "📈 Monitoring"])

# ----------------------------------------------------------
# TAB 1: SINGLE EMPLOYEE
# ----------------------------------------------------------
RATING = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very high"}
WLB = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
EDUCATION = {1: "1 - Below college", 2: "2 - College", 3: "3 - Bachelor", 4: "4 - Master", 5: "5 - Doctor"}
PERFORMANCE = {1: "1 - Low", 2: "2 - Good", 3: "3 - Excellent", 4: "4 - Outstanding"}


def num(label, col, hi=None, **kw):
    """Slider limited to the range seen in training (and optionally a logical upper limit)."""
    lo, top, med = ranges[col]["min"], ranges[col]["max"], ranges[col]["median"]
    top = top if hi is None else min(top, hi)
    if top <= lo:
        st.caption(f"{label}: {lo} (fixed by the values above)")
        return lo
    return st.slider(label, lo, top, min(max(med, lo), top), **kw)


def rating(label, col, names):
    options = list(range(ranges[col]["min"], ranges[col]["max"] + 1))
    default = options.index(ranges[col]["median"])
    return st.selectbox(label, options, index=default, format_func=lambda v: names.get(v, str(v)))


def cat(label, col, default=None):
    options = categories[col]
    return st.selectbox(label, options, index=options.index(default) if default in options else 0)


with tab_single:
    t1, t2, t3 = st.tabs(["Role & pay", "Career & tenure", "Engagement & surveys"])
    with t1:
        c1, c2, c3 = st.columns(3)
        with c1:
            age = num("Age", "Age")
            gender = cat("Gender", "Gender")
            marital = cat("Marital status", "MaritalStatus", "Married")
            distance = num("Distance from home (km)", "DistanceFromHome")
        with c2:
            department = cat("Department", "Department", "Research & Development")
            job_role = cat("Job role", "JobRole", "Research Scientist")
            job_level = rating("Job level (1 = entry, 5 = executive)", "JobLevel", {})
            business_travel = cat("Business travel", "BusinessTravel", "Travel_Rarely")
        with c3:
            income = st.number_input("Monthly income ($)", ranges["MonthlyIncome"]["min"],
                                     ranges["MonthlyIncome"]["max"], ranges["MonthlyIncome"]["median"], step=100)
            hike = num("Last salary hike (%)", "PercentSalaryHike")
            stock = rating("Stock option level (0 = none)", "StockOptionLevel", {})
            overtime = st.radio("Works overtime?", categories["OverTime"], horizontal=True)
    with t2:
        c1, c2, c3 = st.columns(3)
        with c1:
            total_years = num("Total working years", "TotalWorkingYears", hi=max(age - 16, 0))
            companies = num("Number of companies worked for", "NumCompaniesWorked")
            education = rating("Education", "Education", EDUCATION)
            edu_field = cat("Education field", "EducationField", "Life Sciences")
        with c2:
            years_company = num("Years at this company", "YearsAtCompany", hi=total_years)
            years_role = num("Years in current role", "YearsInCurrentRole", hi=years_company)
            years_promo = num("Years since last promotion", "YearsSinceLastPromotion", hi=years_company)
            years_manager = num("Years with current manager", "YearsWithCurrManager", hi=years_company)
        with c3:
            training = num("Trainings last year", "TrainingTimesLastYear")
            performance = rating("Performance rating", "PerformanceRating", PERFORMANCE)
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            job_sat = rating("Job satisfaction", "JobSatisfaction", RATING)
            env_sat = rating("Environment satisfaction", "EnvironmentSatisfaction", RATING)
            rel_sat = rating("Relationship satisfaction", "RelationshipSatisfaction", RATING)
        with c2:
            wlb = rating("Work-life balance", "WorkLifeBalance", WLB)
            involvement = rating("Job involvement", "JobInvolvement", RATING)

    employee = {
        "Age": age, "BusinessTravel": business_travel, "Department": department,
        "DistanceFromHome": distance, "Education": education, "EducationField": edu_field,
        "EnvironmentSatisfaction": env_sat, "Gender": gender, "JobInvolvement": involvement,
        "JobLevel": job_level, "JobRole": job_role, "JobSatisfaction": job_sat,
        "MaritalStatus": marital, "MonthlyIncome": int(income), "NumCompaniesWorked": companies,
        "OverTime": overtime, "PercentSalaryHike": hike, "PerformanceRating": performance,
        "RelationshipSatisfaction": rel_sat, "StockOptionLevel": stock,
        "TotalWorkingYears": total_years, "TrainingTimesLastYear": training,
        "WorkLifeBalance": wlb, "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role, "YearsSinceLastPromotion": years_promo,
        "YearsWithCurrManager": years_manager,
    }

    st.divider()
    if st.button("🔍 Assess attrition risk", type="primary", width="stretch"):
        start = time.perf_counter()
        prob = float(predict(pd.DataFrame([employee]))[0])
        latency_ms = (time.perf_counter() - start) * 1000
        pred = int(prob >= threshold)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            append_csv(LOG_PATH, {"timestamp_utc": stamp, "probability": round(prob, 4), "prediction": pred,
                                  "threshold": threshold, "latency_ms": round(latency_ms, 2), **employee})
        except OSError:
            st.warning("Prediction could not be written to the monitoring log.")
        st.session_state["last"] = {"time": stamp, "prob": prob, "pred": pred, "latency": latency_ms}

    last = st.session_state.get("last")
    if last:
        prob, pred = last["prob"], last["pred"]
        feats = add_features(pd.DataFrame([employee])).iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Attrition probability", f"{prob:.1%}",
                  delta=f"{(prob - threshold) * 100:+.1f} pts vs threshold", delta_color="inverse")
        k2.metric("Overall satisfaction", f"{feats['Overall_Satisfaction']:.2f} / 4")
        k3.metric("Overtime strain", f"{int(feats['Overtime_Strain'])} / 4",
                  help="Overtime (0/1) x (5 - work-life balance). The model's strongest risk factor.")
        k4.metric("Response time", f"{last['latency']:.0f} ms")
        st.progress(min(max(prob, 0.0), 1.0))

        if pred:
            st.error(f"⚠️ **HIGH ATTRITION RISK** — {prob:.1%} is at or above the {threshold:.0%} threshold.")
            actions = []
            if employee["OverTime"] == "Yes":
                actions.append("Review overtime and workload; agree a plan to reduce it.")
            if employee["WorkLifeBalance"] <= 2 or feats["Overall_Satisfaction"] < 2.5:
                actions.append("Hold a stay interview to understand low satisfaction / work-life balance.")
            if employee["YearsSinceLastPromotion"] >= 3:
                actions.append("Discuss career path: no promotion for 3+ years.")
            if employee["StockOptionLevel"] == 0:
                actions.append("Check eligibility for stock options or other retention incentives.")
            if employee["BusinessTravel"] == "Travel_Frequently":
                actions.append("Review the amount of business travel.")
            actions = actions or ["Schedule a 1-on-1 with the manager to discuss engagement and plans."]
            st.markdown("**Suggested HR actions:**\n" + "\n".join(f"- {a}" for a in actions))
        else:
            st.success(f"✅ **LOW ATTRITION RISK** — {prob:.1%} is below the {threshold:.0%} threshold. "
                       "Continue normal engagement check-ins.")

        with st.expander("📝 Give feedback on this prediction (used to monitor real-world accuracy)"):
            outcome = st.radio("What actually happened to this employee?", ["Stayed", "Left"], horizontal=True)
            comment = st.text_input("Comment (optional)")
            if st.button("Save feedback"):
                try:
                    append_csv(FEEDBACK_PATH, {
                        "feedback_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "prediction_time_utc": last["time"], "probability": round(prob, 4),
                        "predicted_leave": pred, "actually_left": int(outcome == "Left"), "comment": comment})
                    st.success("Feedback saved.")
                except OSError:
                    st.warning("Feedback could not be saved.")

# ----------------------------------------------------------
# TAB 2: BATCH SCORING (integration with an HR system export)
# ----------------------------------------------------------
with tab_batch:
    st.markdown("Upload a CSV export of employees with the same columns as the IBM dataset "
                "(extra columns are ignored). Every employee gets a risk score; download the ranked list.")
    upload = st.file_uploader("Employee CSV", type="csv")
    if upload is not None:
        batch = pd.read_csv(upload)
        missing = [c for c in RAW_FEATURES if c not in batch.columns]
        if missing:
            st.error(f"These required columns are missing: {missing}")
        else:
            start = time.perf_counter()
            probs = predict(batch)
            elapsed = (time.perf_counter() - start) * 1000
            result = batch.copy()
            result.insert(0, "Attrition_Probability", probs.round(4))
            result.insert(1, "Risk", np.where(probs >= threshold, "HIGH", "LOW"))
            result = result.sort_values("Attrition_Probability", ascending=False)

            c1, c2, c3 = st.columns(3)
            c1.metric("Employees scored", len(result))
            c2.metric("Flagged high risk", f"{(probs >= threshold).sum()} ({(probs >= threshold).mean():.0%})")
            c3.metric("Scoring time", f"{elapsed:.0f} ms")
            st.dataframe(result, width="stretch", height=350)
            st.download_button("⬇️ Download scored list", result.to_csv(index=False),
                               "attrition_scores.csv", "text/csv")

            if "Attrition" in batch.columns:
                from sklearn.metrics import f1_score, recall_score, roc_auc_score
                actual = batch["Attrition"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
                if actual.notna().all() and actual.nunique() == 2:
                    st.subheader("Performance on this file (it contains real outcomes)")
                    flags = (probs >= threshold).astype(int)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ROC-AUC", f"{roc_auc_score(actual, probs):.3f}")
                    c2.metric("Recall", f"{recall_score(actual, flags):.0%}")
                    c3.metric("F1", f"{f1_score(actual, flags):.3f}")

            st.subheader("Data drift check for this file")
            st.dataframe(drift_table(batch), width="stretch")

# ----------------------------------------------------------
# TAB 3: MONITORING
# ----------------------------------------------------------
with tab_monitor:
    st.markdown("Every single-employee prediction is logged with its inputs and response time. "
                "This tab summarises the log so we can watch speed, data drift and real-world accuracy.")
    if not LOG_PATH.exists():
        st.info("No predictions logged yet. Score an employee in the first tab.")
    else:
        log = pd.read_csv(LOG_PATH)
        expected = bundle["train_stats"]["high_risk_rate"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictions logged", len(log))
        c2.metric("Median latency", f"{log['latency_ms'].median():.0f} ms")
        c3.metric("95th pct latency", f"{log['latency_ms'].quantile(0.95):.0f} ms")
        c4.metric("Flagged high risk", f"{log['prediction'].mean():.0%}",
                  delta=f"{(log['prediction'].mean() - expected) * 100:+.0f} pts vs training ({expected:.0%})",
                  delta_color="off")

        st.subheader("Response latency over time")
        st.line_chart(log["latency_ms"].reset_index(drop=True))

        st.subheader("Data drift (logged inputs vs training data)")
        if len(log) < 30:
            st.caption(f"Based on {len(log)} predictions — drift results become meaningful after about 30.")
        st.dataframe(drift_table(log), width="stretch")

        st.subheader("Recent predictions")
        st.dataframe(log.tail(20).iloc[::-1], width="stretch")
        st.download_button("⬇️ Download prediction log", log.to_csv(index=False), "prediction_log.csv")

    st.subheader("Feedback from HR (real outcomes)")
    if not FEEDBACK_PATH.exists():
        st.info("No feedback yet. After a prediction, use 'Give feedback on this prediction'.")
    else:
        fb = pd.read_csv(FEEDBACK_PATH)
        correct = (fb["predicted_leave"] == fb["actually_left"]).mean()
        leavers = fb[fb["actually_left"] == 1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Feedback records", len(fb))
        c2.metric("Predictions correct", f"{correct:.0%}")
        c3.metric("Leavers caught (recall)",
                  f"{leavers['predicted_leave'].mean():.0%}" if len(leavers) else "n/a")
        st.caption("If recall from feedback falls below 70%, or drift is flagged, retrain the model "
                   "by re-running both notebooks with the new data.")
        st.dataframe(fb.tail(20).iloc[::-1], width="stretch")

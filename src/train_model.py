import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_pipeline import load_data, preprocess, build_master_dataset


if __name__ == "__main__":

    # Load engineered dataset
    accounts, subscriptions, feature_usage, support_tickets, churn_events = load_data()
    accounts, subscriptions, feature_usage, support_tickets, churn_events = preprocess(
        accounts, subscriptions, feature_usage, support_tickets, churn_events
    )

    master = build_master_dataset(
        accounts, subscriptions, feature_usage, support_tickets, churn_events
    )

    # ----------------------------
    # Feature Selection
    # ----------------------------
    
    feature_cols = master.drop(columns=[
    "account_id",
    "account_name",
    "churn_flag",   # REMOVE TARGET
    "churned"       # remove old derived label
    ])

    # Remove datetime columns
    feature_cols = feature_cols.select_dtypes(exclude=["datetime64[ns]"])

    # One-hot encode categorical features
    X = pd.get_dummies(feature_cols, drop_first=True)

    y = master["churn_flag"]

    # ----------------------------
    # Train-Test Split
    # ----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ----------------------------
    # Logistic Regression (Baseline)
    # ----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    log_probs = log_model.predict_proba(X_test_scaled)[:, 1]
    log_auc = roc_auc_score(y_test, log_probs)

    print("Logistic Regression AUC:", round(log_auc, 4))

    # ----------------------------
    # Random Forest
    # ----------------------------
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)

    print("Random Forest AUC:", round(rf_auc, 4))

    # ----------------------------
    # XGBoost
    # ----------------------------
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_probs)

    print("XGBoost AUC:", round(xgb_auc, 4))

    # Save best model (we'll assume XGBoost for now)
    joblib.dump(xgb_model, "xgb_churn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
# ----------------------------
# Generate Risk Scores for All Accounts
# ----------------------------

    all_probs = xgb_model.predict_proba(X)[:, 1]

    master["churn_risk_score"] = all_probs

    # Save dataset with predictions
    master.to_csv("master_with_risk.csv", index=False)

    print("Risk scores generated and saved.")
    print("Model saved successfully.")
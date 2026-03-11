import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data")
MODEL_PATH = Path("models")


if __name__ == "__main__":

    print("Loading master dataset...")

    master = pd.read_csv(DATA_PATH / "master_dataset.csv")

    # ----------------------------
    # Feature Selection
    # ----------------------------

    feature_cols = master.drop(columns=[
        "account_id",
        "account_name",
        "churned"   # target
    ], errors="ignore")

    # Remove datetime columns
    feature_cols = feature_cols.select_dtypes(exclude=["datetime64[ns]"])

    # One-hot encode categorical features
    X = pd.get_dummies(feature_cols, drop_first=True)

    y = master["churned"]

    # ----------------------------
    # Train Test Split
    # ----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # ----------------------------
    # Logistic Regression
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
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)

    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_probs)

    print("XGBoost AUC:", round(xgb_auc, 4))

    # ----------------------------
    # Save Best Model
    # ----------------------------

    MODEL_PATH.mkdir(exist_ok=True)

    joblib.dump(xgb_model, MODEL_PATH / "churn_model.pkl")
    joblib.dump(scaler, MODEL_PATH / "scaler.pkl")

    print("Model saved.")

    # ----------------------------
    # Generate Risk Scores
    # ----------------------------

    all_probs = xgb_model.predict_proba(X)[:, 1]

    master["churn_risk_score"] = all_probs

    master.to_csv("data/master_dataset.csv", index=False)

    print("Risk scores added to master dataset.")
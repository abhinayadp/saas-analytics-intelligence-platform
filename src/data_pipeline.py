import pandas as pd
from pathlib import Path

DATA_PATH = Path("../data")

def load_data():
    accounts = pd.read_csv(DATA_PATH / "ravenstack_accounts.csv")
    subscriptions = pd.read_csv(DATA_PATH / "ravenstack_subscriptions.csv")
    feature_usage = pd.read_csv(DATA_PATH / "ravenstack_feature_usage.csv")
    support_tickets = pd.read_csv(DATA_PATH / "ravenstack_support_tickets.csv")
    churn_events = pd.read_csv(DATA_PATH / "ravenstack_churn_events.csv")

    return accounts, subscriptions, feature_usage, support_tickets, churn_events


def preprocess(accounts, subscriptions, feature_usage, support_tickets, churn_events):

    # Convert date columns
    date_cols = {
        "accounts": ["signup_date"],
        "subscriptions": ["start_date", "end_date"],
        "support_tickets": ["created_at", "resolved_at"],
        "churn_events": ["churn_date"]
    }

    for col in date_cols["accounts"]:
        if col in accounts.columns:
            accounts[col] = pd.to_datetime(accounts[col], errors="coerce")

    for col in date_cols["subscriptions"]:
        if col in subscriptions.columns:
            subscriptions[col] = pd.to_datetime(subscriptions[col], errors="coerce")

    for col in date_cols["support_tickets"]:
        if col in support_tickets.columns:
            support_tickets[col] = pd.to_datetime(support_tickets[col], errors="coerce")

    for col in date_cols["churn_events"]:
        if col in churn_events.columns:
            churn_events[col] = pd.to_datetime(churn_events[col], errors="coerce")

    return accounts, subscriptions, feature_usage, support_tickets, churn_events

def build_master_dataset(accounts, subscriptions, feature_usage, support_tickets, churn_events):

    # --- Subscription Aggregation ---
    subs_agg = subscriptions.groupby("account_id").agg(
        total_subscriptions=("subscription_id", "count"),
        avg_mrr=("mrr_amount", "mean"),
        total_upgrades=("upgrade_flag", "sum"),
        total_downgrades=("downgrade_flag", "sum")
    ).reset_index()

    # --- Feature Usage Aggregation ---
    usage_agg = feature_usage.groupby("subscription_id").agg(
        total_usage=("usage_count", "sum"),
        total_usage_duration=("usage_duration_secs", "sum"),
        total_errors=("error_count", "sum"),
        beta_feature_usage=("is_beta_feature", "sum")
    ).reset_index()

    usage_agg = usage_agg.merge(
        subscriptions[["subscription_id", "account_id"]],
        on="subscription_id",
        how="left"
    )

    usage_agg = usage_agg.groupby("account_id").agg(
        total_usage=("total_usage", "sum"),
        total_usage_duration=("total_usage_duration", "sum"),
        total_errors=("total_errors", "sum"),
        beta_feature_usage=("beta_feature_usage", "sum")
    ).reset_index()

    # --- Support Aggregation ---
    support_agg = support_tickets.groupby("account_id").agg(
        total_tickets=("ticket_id", "count"),
        avg_resolution_time=("resolution_time_hours", "mean"),
        avg_first_response=("first_response_time_minutes", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        escalation_count=("escalation_flag", "sum")
    ).reset_index()

    # --- Churn Label ---
    churn_agg = churn_events.groupby("account_id").size().reset_index(name="churn_events")
    churn_agg["churned"] = 1

    # --- Merge Everything ---
    master = accounts.merge(subs_agg, on="account_id", how="left")
    master = master.merge(usage_agg, on="account_id", how="left")
    master = master.merge(support_agg, on="account_id", how="left")
    master = master.merge(churn_agg[["account_id", "churned"]], on="account_id", how="left")

    master["churned"] = master["churned"].fillna(0)

    numeric_cols = master.select_dtypes(include=["float64", "int64"]).columns
    master[numeric_cols] = master[numeric_cols].fillna(0)

    master["usage_per_subscription"] = master["total_usage"] / (master["total_subscriptions"] + 1)
    master["tickets_per_subscription"] = master["total_tickets"] / (master["total_subscriptions"] + 1)
    master["error_rate"] = master["total_errors"] / (master["total_usage"] + 1)
    master["mrr_per_seat"] = master["avg_mrr"] / (master["seats"] + 1)

    master["usage_per_subscription"] = master["usage_per_subscription"].fillna(0)
    master["tickets_per_subscription"] = master["tickets_per_subscription"].fillna(0)
    master["error_rate"] = master["error_rate"].fillna(0)
    master["mrr_per_seat"] = master["mrr_per_seat"].fillna(0)
    return master


if __name__ == "__main__":
    accounts, subscriptions, feature_usage, support_tickets, churn_events = load_data()

    accounts, subscriptions, feature_usage, support_tickets, churn_events = preprocess(
        accounts, subscriptions, feature_usage, support_tickets, churn_events
    )

    master = build_master_dataset(
        accounts, subscriptions, feature_usage, support_tickets, churn_events
    )

    print("Master Dataset Shape:", master.shape)
    print(master.head())

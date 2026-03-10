import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="SaaS Analytics Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("data/master_dataset.csv")

st.title("🚀 SaaS Analytics Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

industry_filter = st.sidebar.multiselect(
    "Select Industry",
    options=df["industry"].unique(),
    default=df["industry"].unique()
)

plan_filter = st.sidebar.multiselect(
    "Select Plan Tier",
    options=df["plan_tier"].unique(),
    default=df["plan_tier"].unique()
)

filtered_df = df[
    (df["industry"].isin(industry_filter)) &
    (df["plan_tier"].isin(plan_filter))
].copy()

# -----------------------------
# KPIs
# -----------------------------

total_accounts = filtered_df.shape[0]
churn_rate = filtered_df["churned"].mean() * 100
avg_mrr = filtered_df["avg_mrr"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Accounts", total_accounts)
col2.metric("Churn Rate (%)", f"{churn_rate:.2f}")
col3.metric("Average MRR", f"${avg_mrr:.2f}")

st.divider()

# -----------------------------
# Customer Distribution
# -----------------------------

st.subheader("Customer Distribution")

status_counts = (
    filtered_df["churned"]
    .value_counts()
    .reset_index()
)

status_counts.columns = ["churned", "count"]
status_counts["churned"] = status_counts["churned"].map({0: "Active", 1: "Churned"})

fig_counts = px.bar(
    status_counts,
    x="churned",
    y="count",
    labels={"churned": "Customer Status", "count": "Number of Accounts"},
    title="Active vs Churned Customers"
)

st.plotly_chart(fig_counts, use_container_width=True)

# -----------------------------
# Top Risk Accounts
# -----------------------------

st.subheader("Top 10 High-Risk Accounts")

top_risk = (
    filtered_df.sort_values("churn_risk_score", ascending=False)
    .head(10)[["account_name", "industry", "plan_tier", "churn_risk_score"]]
)

st.dataframe(top_risk)

# -----------------------------
# Risk Score Distribution
# -----------------------------

st.subheader("Churn Risk Distribution")

fig_risk = px.histogram(
    filtered_df,
    x="churn_risk_score",
    nbins=20,
    title="Distribution of Churn Risk Scores"
)

st.plotly_chart(fig_risk, use_container_width=True)

# -----------------------------
# Risk Segments
# -----------------------------

filtered_df["risk_segment"] = pd.cut(
    filtered_df["churn_risk_score"],
    bins=[0, 0.3, 0.6, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

st.subheader("Risk Segment Distribution")

risk_counts = filtered_df["risk_segment"].value_counts().reset_index()
risk_counts.columns = ["Risk Segment", "Count"]

fig_seg = px.bar(
    risk_counts,
    x="Risk Segment",
    y="Count",
    title="Customer Risk Segments"
)

st.plotly_chart(fig_seg, use_container_width=True)

# -----------------------------
# Churn by Industry
# -----------------------------

st.subheader("Churn Rate by Industry")

industry_churn = (
    filtered_df.groupby("industry")["churned"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

fig_industry = px.bar(
    industry_churn,
    x="industry",
    y="churned",
    title="Churn Rate by Industry"
)

st.plotly_chart(fig_industry, use_container_width=True)

# -----------------------------
# Feature Usage vs Churn
# -----------------------------

st.subheader("Feature Usage vs Churn")

usage_churn = (
    filtered_df.groupby("churned")["total_usage"]
    .mean()
    .reset_index()
)

usage_churn["churned"] = usage_churn["churned"].map({0: "Active", 1: "Churned"})

fig_usage = px.bar(
    usage_churn,
    x="churned",
    y="total_usage",
    title="Average Feature Usage by Customer Status"
)

st.plotly_chart(fig_usage, use_container_width=True)

# -----------------------------
# Support Tickets vs Churn
# -----------------------------

support_churn = (
    filtered_df.groupby("churned")["total_tickets"]
    .mean()
    .reset_index()
)

support_churn["churned"] = support_churn["churned"].map({0: "Active", 1: "Churned"})

fig_support = px.bar(
    support_churn,
    x="churned",
    y="total_tickets",
    title="Average Support Tickets by Customer Status"
)

st.plotly_chart(fig_support, use_container_width=True)

# -----------------------------
# A/B Test Simulation
# -----------------------------

st.divider()
st.subheader("🧪 Retention Campaign A/B Test Simulation")

high_risk_df = filtered_df[filtered_df["churn_risk_score"] > 0.6].copy()

if len(high_risk_df) > 20:

    high_risk_df["group"] = np.random.choice(
        ["Control", "Treatment"],
        size=len(high_risk_df)
    )

    high_risk_df["simulated_churn"] = high_risk_df["churned"]

    treatment_mask = high_risk_df["group"] == "Treatment"

    high_risk_df.loc[treatment_mask, "simulated_churn"] = (
        high_risk_df.loc[treatment_mask, "churned"] * 0.8
    )

    experiment_results = (
        high_risk_df.groupby("group")["simulated_churn"]
        .mean()
        .reset_index()
    )

    st.write("### Experiment Results")
    st.dataframe(experiment_results)

    control_rate = experiment_results[
        experiment_results["group"] == "Control"
    ]["simulated_churn"].values[0]

    treatment_rate = experiment_results[
        experiment_results["group"] == "Treatment"
    ]["simulated_churn"].values[0]

    uplift = control_rate - treatment_rate

    st.metric("Churn Reduction (Absolute Uplift)", f"{uplift:.4f}")

    avg_mrr = high_risk_df["avg_mrr"].mean()
    revenue_saved = uplift * len(high_risk_df) * avg_mrr

    st.metric("Estimated Monthly Revenue Saved", f"${revenue_saved:,.2f}")

else:
    st.info("Not enough high-risk users to simulate experiment.")
import os
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import plotly.express as px

load_dotenv()

st.set_page_config(page_title="SaaS Analytics Dashboard", layout="wide")

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["saas_analytics"]
collection = db["accounts_master"]

# Load data from MongoDB
data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

st.title("🚀 SaaS Analytics Dashboard")

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
]

# --- KPIs ---
total_accounts = filtered_df.shape[0]
churn_rate = filtered_df["churned"].mean() * 100
avg_mrr = filtered_df["avg_mrr"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Accounts", total_accounts)
col2.metric("Churn Rate (%)", f"{churn_rate:.2f}")
col3.metric("Average MRR", f"${avg_mrr:.2f}")

st.divider()

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
    labels={
        "churned": "Customer Status",
        "count": "Number of Accounts"
    },
    title="Number of Active vs Churned Accounts"
)

st.plotly_chart(fig_counts, use_container_width=True)

st.subheader("Top 10 High-Risk Accounts")

top_risk = (
    filtered_df.sort_values("churn_risk_score", ascending=False)
    .head(10)[["account_name", "industry", "plan_tier", "churn_risk_score"]]
)

st.dataframe(top_risk)


st.subheader("Churn Risk Distribution")

# churn risk distribution
fig_risk = px.histogram(
    filtered_df,
    x="churn_risk_score",
    nbins=20,
    title="Distribution of Churn Risk Scores"
)

st.plotly_chart(fig_risk, use_container_width=True)

#risk segments
filtered_df["risk_segment"] = pd.cut(
    filtered_df["churn_risk_score"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

st.subheader("Risk Segment Distribution")

risk_counts = filtered_df["risk_segment"].value_counts().reset_index()
risk_counts.columns = ["Risk Segment", "Count"]

fig_seg = px.bar(risk_counts, x="Risk



 Segment", y="Count",
                 title="Customer Risk Segments")

st.plotly_chart(fig_seg, use_container_width=True)

# --- Churn by Industry ---
st.subheader("Churn Rate by Industry")

industry_churn = (
    filtered_df.groupby("industry")["churned"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

st.bar_chart(industry_churn.set_index("industry"))

# --- Feature Usage vs Churn ---
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
    labels={
        "churned": "Customer Status",
        "total_usage": "Average Feature Usage"
    },
    title="Average Feature Usage by Customer Status"
)

st.plotly_chart(fig_usage, use_container_width=True)


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
    labels={
        "churned": "Customer Status",
        "total_tickets": "Average Number of Support Tickets"
    },
    title="Average Support Tickets by Customer Status"
)

st.plotly_chart(fig_support, use_container_width=True)

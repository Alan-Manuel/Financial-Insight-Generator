import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Financial Insight Generator", layout="wide")
st.title("Financial Insight Generator")
st.markdown("Upload a CSV of your transactions to receive automated insights and interactive visual analytics (Plotly).")

with st.expander("How to use this tool", expanded=True):
    st.markdown(
        """
### Quick start
1. Upload a CSV that has at least an **Amount** column.
2. (Optional) Include **Date**, **Category**, and **Merchant** columns for richer charts.
3. Click **Generate Insights**.

### Columns
- Required: `Amount`
- Optional: `Date`, `Category`, `Merchant`
"""
    )


# -----------------------------
# Helpers
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def parse_amount(series: pd.Series) -> pd.Series:
    # handles "$1,234.50" and "(45.00)"
    s = series.astype(str).str.strip()
    s = s.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    s = s.str.replace(r"[\$,]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def cap_series(series: pd.Series, q: float) -> pd.Series:
    cap_val = series.quantile(q)
    return np.minimum(series, cap_val)


# -----------------------------
# Analysis
# -----------------------------
def analyze_transactions(df: pd.DataFrame, contamination: float, n_clusters: int) -> tuple[list[str], pd.DataFrame]:
    df = df.copy()

    # Isolation Forest anomaly detection (amount only)
    iso = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly_flag"] = (iso.fit_predict(df[["Amount"]]) == -1).astype(int)  # 1 = anomaly

    # KMeans clustering (scale amount)
    scaler = StandardScaler()
    amt_scaled = scaler.fit_transform(df[["Amount"]].values)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(amt_scaled)

    # Insights
    total_spent = float(df["Amount"].sum())
    avg_spent = float(df["Amount"].mean())
    anomaly_count = int(df["anomaly_flag"].sum())
    anomaly_rate = float(df["anomaly_flag"].mean() * 100)

    high_spend = df[df["Amount"] > df["Amount"].mean() + 2 * df["Amount"].std()]

    insights = [
        f"Total Spending: ${total_spent:,.2f}",
        f"Average Transaction: ${avg_spent:,.2f}",
        f"Anomalies Flagged: {anomaly_count:,} ({anomaly_rate:.2f}%)",
    ]
    if not high_spend.empty:
        insights.append(f"{len(high_spend):,} unusually high-value transactions detected (> mean + 2×std).")

    return insights, df


# -----------------------------
# Sidebar Controls (makes it feel like a real dashboard)
# -----------------------------
st.sidebar.header("Controls")

contamination = st.sidebar.slider(
    "Anomaly sensitivity (IsolationForest contamination)",
    0.01, 0.20, 0.05, 0.01
)

n_clusters = st.sidebar.slider(
    "Number of clusters (KMeans)",
    2, 8, 3, 1
)

cap_toggle = st.sidebar.checkbox("Cap extreme values in charts", value=True)
cap_q = st.sidebar.slider("Cap quantile", 0.90, 0.999, 0.98, 0.001)

chart_choices = st.sidebar.multiselect(
    "Charts to display",
    [
        "Histogram",
        "Boxplot",
        "Clusters (Scatter)",
        "Anomalies Table",
        "Top Categories",
        "Top Merchants",
        "Spending Over Time (if Date exists)",
    ],
    default=["Histogram", "Boxplot", "Clusters (Scatter)", "Anomalies Table"]
)

top_n = st.sidebar.slider("Top-N (categories/merchants)", 5, 30, 10, 1)


# -----------------------------
# Upload UI
# -----------------------------
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin. Minimum requirement: an `Amount` column.")
    st.stop()


# -----------------------------
# Main workflow
# -----------------------------
try:
    df = pd.read_csv(uploaded_file)
    df = normalize_columns(df)

    if "Amount" not in df.columns and "amount" in df.columns:
        df = df.rename(columns={"amount": "Amount"})

    if "Amount" not in df.columns:
        st.error("Your CSV must contain an 'Amount' column (or 'amount').")
        st.stop()

    # parse numeric amount
    df["Amount"] = parse_amount(df["Amount"])
    df = df.dropna(subset=["Amount"]).copy()

    # optional date parsing (if provided)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"**Total Rows:** {df.shape[0]:,} | **Columns:** {df.shape[1]}")

    if st.button("Generate Insights", type="primary"):
        with st.spinner("Analyzing your transactions..."):
            insights, df_out = analyze_transactions(df, contamination=contamination, n_clusters=n_clusters)

        # add capped amount for plots only
        df_out["Amount_plot"] = cap_series(df_out["Amount"], cap_q) if cap_toggle else df_out["Amount"]

        # Summary metrics
        st.subheader("Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{df_out.shape[0]:,}")
        col2.metric("Total Spending", f"${df_out['Amount'].sum():,.2f}")
        col3.metric("Avg Transaction", f"${df_out['Amount'].mean():,.2f}")
        col4.metric("Anomalies", f"{int(df_out['anomaly_flag'].sum()):,}")

        # Insights
        st.subheader("Key Insights")
        st.info("Here's what we found from your data:")
        for insight in insights:
            st.markdown(f"- {insight}")

        # Charts
        st.subheader("Interactive Charts")

        if "Histogram" in chart_choices:
            fig = px.histogram(df_out, x="Amount_plot", nbins=40, title="Distribution of Transaction Amounts")
            st.plotly_chart(fig, use_container_width=True)

        if "Boxplot" in chart_choices:
            fig = px.box(df_out, x="Amount_plot", points="outliers", title="Transaction Value Spread (Boxplot)")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        if "Clusters (Scatter)" in chart_choices:
            tmp = df_out.reset_index(drop=True)
            fig = px.scatter(
                tmp,
                x=tmp.index,
                y="Amount_plot",
                color="cluster",
                symbol=tmp["anomaly_flag"].map({0: "Normal", 1: "Anomaly"}),
                title="KMeans Clustering (color) + Anomaly Flag (symbol)",
                hover_data=[c for c in ["Date", "Category", "Merchant", "Amount"] if c in tmp.columns],
                opacity=0.75,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        if "Anomalies Table" in chart_choices:
            st.markdown("### Flagged Anomalies (Top 50 by Amount)")
            anomalies = df_out[df_out["anomaly_flag"] == 1].copy().sort_values("Amount", ascending=False).head(50)
            if anomalies.empty:
                st.info("No anomalies were flagged at this sensitivity.")
            else:
                cols = [c for c in ["Date", "Category", "Merchant", "Amount", "cluster"] if c in anomalies.columns]
                st.dataframe(anomalies[cols], use_container_width=True)

        if "Top Categories" in chart_choices:
            if "Category" in df_out.columns:
                cat = df_out.groupby("Category")["Amount"].sum().sort_values(ascending=False).head(top_n).reset_index()
                fig = px.bar(cat, x="Category", y="Amount", title=f"Top {top_n} Categories by Spend")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Top Categories chart requires a `Category` column.")

        if "Top Merchants" in chart_choices:
            if "Merchant" in df_out.columns:
                mer = df_out.groupby("Merchant")["Amount"].sum().sort_values(ascending=False).head(top_n).reset_index()
                fig = px.bar(mer, x="Merchant", y="Amount", title=f"Top {top_n} Merchants by Spend")
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Top Merchants chart requires a `Merchant` or `Description` column.")

        if "Spending Over Time (if Date exists)" in chart_choices:
            if "Date" in df_out.columns and df_out["Date"].notna().any():
                ts = df_out.dropna(subset=["Date"]).copy()
                ts = ts.sort_values("Date")
                daily = ts.groupby(ts["Date"].dt.date)["Amount"].sum().reset_index()
                daily.columns = ["Date", "DailySpend"]
                fig = px.line(daily, x="Date", y="DailySpend", title="Daily Spend Over Time")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Spending Over Time requires a parseable `Date` column.")

except Exception as e:
    st.error(f"Error processing file: {e}")

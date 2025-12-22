import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Financial Insight Generator", layout="centered")
st.title("Financial Insight Generator")
st.markdown("Upload a CSV of your transactions to receive automated insights and visual analytics.")

with st.expander("What your CSV needs", expanded=True):
    st.markdown(
        """
**Required**
- `Amount` column (numeric)

**Tips**
- Works even if your Amounts look like `$1,234.56` or `(45.00)`
- You can optionally add other columns like Date/Category/Merchant (not required)
"""
    )

# -----------------------------
# Helpers
# -----------------------------
def parse_amount(series: pd.Series) -> pd.Series:
    """Convert strings like '$1,234.56' or '(45.00)' into numbers."""
    s = series.astype(str).str.strip()
    s = s.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    s = s.str.replace(r"[\$,]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def cap_for_chart(series: pd.Series, q: float) -> pd.Series:
    """Cap extreme values for plots only (winsorize)."""
    cap = series.quantile(q)
    return np.minimum(series, cap)

# -----------------------------
# Analysis
# -----------------------------
def analyze_transactions(df: pd.DataFrame, contamination: float, n_clusters: int):
    df = df.copy()

    # Anomaly detection
    iso = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly_flag"] = (iso.fit_predict(df[["Amount"]]) == -1).astype(int)

    # Clustering (scale Amount so KMeans behaves better)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[["Amount"]].values)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(X)

    # Insights
    total_spent = df["Amount"].sum()
    avg_spent = df["Amount"].mean()
    anomaly_count = int(df["anomaly_flag"].sum())
    anomaly_rate = float(df["anomaly_flag"].mean() * 100)

    high_spend = df[df["Amount"] > df["Amount"].mean() + 2 * df["Amount"].std()]

    insights = []
    insights.append(f"Total Spending: ${total_spent:,.2f}")
    insights.append(f"Average Transaction: ${avg_spent:,.2f}")
    insights.append(f"Anomalies Flagged: {anomaly_count:,} ({anomaly_rate:.2f}%)")
    if not high_spend.empty:
        insights.append(f"{len(high_spend):,} unusually high-value transactions (> mean + 2×std).")
    insights.append("Tip: Review anomalies + top transactions weekly to catch unusual spending early.")

    return insights, df

# -----------------------------
# Upload UI
# -----------------------------
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)

    if "Amount" not in df.columns:
        st.error("The uploaded file must contain an 'Amount' column.")
        st.stop()

    # Clean Amount
    df["Amount"] = parse_amount(df["Amount"])
    df = df.dropna(subset=["Amount"]).copy()

    if df.empty:
        st.error("No valid numeric Amount values found after parsing.")
        st.stop()

    # Controls to make charts “look better”
    st.sidebar.header("Chart Controls")
    cap_toggle = st.sidebar.checkbox("Cap extreme values (recommended)", value=True)
    cap_q = st.sidebar.slider("Cap quantile", 0.90, 0.999, 0.98, 0.001)
    contamination = st.sidebar.slider("Anomaly sensitivity", 0.01, 0.20, 0.05, 0.01)
    n_clusters = st.sidebar.slider("Clusters (KMeans)", 2, 8, 3, 1)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(10))

    st.write(f"**Total Rows:** {df.shape[0]:,} | **Columns:** {df.shape[1]}")

    if st.button("Generate Insights", type="primary"):
        with st.spinner("Analyzing your transactions..."):
            insights, df_out = analyze_transactions(df, contamination=contamination, n_clusters=n_clusters)

        # Summary metrics
        st.subheader("Summary Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{df_out.shape[0]:,}")
        c2.metric("Total Spending", f"${df_out['Amount'].sum():,.2f}")
        c3.metric("Avg Amount", f"${df_out['Amount'].mean():,.2f}")
        c4.metric("Anomalies", f"{int(df_out['anomaly_flag'].sum()):,}")

        # Key insights
        st.subheader("Key Insights")
        st.info("Here's what we found:")
        for x in insights:
            st.markdown(f"- {x}")

        # Make a chart-friendly amount column
        if cap_toggle:
            df_out["Amount_plot"] = cap_for_chart(df_out["Amount"], cap_q)
            chart_label = f"Amount (capped at {cap_q:.3f} quantile)"
        else:
            df_out["Amount_plot"] = df_out["Amount"]
            chart_label = "Amount"

        # Chart 1: Histogram (cleaner defaults)
        st.subheader("Distribution of Transaction Amounts")
        fig1, ax1 = plt.subplots()
        ax1.hist(df_out["Amount_plot"], bins=35, edgecolor="black")
        ax1.set_xlabel(chart_label)
        ax1.set_ylabel("Count")
        ax1.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig1)

        # Chart 2: Boxplot (cleaner)
        st.subheader("Transaction Value Spread (Boxplot)")
        fig2, ax2 = plt.subplots(figsize=(8, 2))
        ax2.boxplot(df_out["Amount_plot"], vert=False, notch=True)
        ax2.set_xlabel(chart_label)
        ax2.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig2)

        # Chart 3: Cluster scatter (cluster + anomaly markers)
        st.subheader("Transaction Clustering (KMeans)")
        fig3, ax3 = plt.subplots(figsize=(9, 4))

        colors = plt.cm.tab10(df_out["cluster"] % 10)
        ax3.scatter(
            df_out.index,
            df_out["Amount_plot"],
            c=colors,
            alpha=0.7,
            s=25
        )

        # highlight anomalies with a different marker outline
        anom = df_out["anomaly_flag"] == 1
        ax3.scatter(
            df_out.index[anom],
            df_out.loc[anom, "Amount_plot"],
            facecolors="none",
            edgecolors="red",
            s=70,
            linewidths=1.5,
            label="Anomaly"
        )

        ax3.set_xlabel("Transaction Index")
        ax3.set_ylabel(chart_label)
        ax3.set_title("Clusters (color) + Anomalies (red outline)")
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.legend(loc="upper right")
        st.pyplot(fig3)

        # Extra: show top transactions + anomalies table
        st.subheader("Top Transactions (by Amount)")
        st.dataframe(df_out.sort_values("Amount", ascending=False).head(20))

        st.subheader("Flagged Anomalies (Top 50)")
        st.dataframe(df_out[df_out["anomaly_flag"] == 1].sort_values("Amount", ascending=False).head(50))

except Exception as e:
    st.error(f"Error processing file: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # optional now (can remove if you want)

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from fpdf import FPDF
import plotly.express as px


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
# PDF helpers (safe wrapping + encoding)
# -----------------------------
def _wrap_hard(text: str, width: int = 60) -> str:
    """Hard-wrap every N characters so FPDF never sees an unbreakable long token."""
    text = str(text)
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))


def _pdf_safe(text: str) -> str:
    """Make text safe for FPDF (wrap + latin-1 replace)."""
    wrapped = _wrap_hard(text, width=60)
    return wrapped.encode("latin-1", "replace").decode("latin-1")


def insights_to_pdf_bytes(title: str, lines: list[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.set_font("Helvetica", style="B", size=14)
    pdf.multi_cell(0, 8, _pdf_safe(title))
    pdf.ln(2)

    pdf.set_font("Helvetica", size=11)
    for line in lines:
        pdf.multi_cell(0, 6, _pdf_safe(f"- {line}"))

    return pdf.output(dest="S").encode("latin-1")


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

        # -----------------------------
        # Plotly Charts
        # -----------------------------
        st.subheader("Interactive Charts (Plotly)")

        # Plotly Histogram
        st.markdown("### Distribution of Transaction Amounts")
        fig_hist = px.histogram(
            df_out,
            x="Amount_plot",
            nbins=35,
            title="Transaction Amount Distribution",
            labels={"Amount_plot": chart_label},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Plotly Boxplot
        st.markdown("### Transaction Value Spread (Boxplot)")
        fig_box = px.box(
            df_out,
            x="Amount_plot",
            points="outliers",
            title="Amount Spread (Capped if enabled)",
            labels={"Amount_plot": chart_label},
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Plotly Scatter: clusters + anomaly flag
        st.markdown("### Clustering (KMeans) + Anomalies")
        tmp = df_out.reset_index(drop=True).copy()
        tmp["status"] = tmp["anomaly_flag"].map({0: "Normal", 1: "Anomaly"})

        fig_scatter = px.scatter(
            tmp,
            x=tmp.index,
            y="Amount_plot",
            color="cluster",
            symbol="status",
            title="Clusters (color) + Anomalies (symbol)",
            labels={"Amount_plot": chart_label, "x": "Transaction Index"},
            hover_data=["Amount", "cluster", "status"],
            opacity=0.75,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # -----------------------------
        # Tables
        # -----------------------------
        st.subheader("Top Transactions (by Amount)")
        st.dataframe(df_out.sort_values("Amount", ascending=False).head(20))

        st.subheader("Flagged Anomalies (Top 50)")
        anomalies_df = df_out[df_out["anomaly_flag"] == 1].sort_values("Amount", ascending=False).head(50)
        st.dataframe(anomalies_df)

        # -----------------------------
        # PDF Export (below anomalies)
        # -----------------------------
        st.subheader("Download Report (PDF)")

        pdf_lines = [
            f"Transactions: {df_out.shape[0]:,}",
            f"Total Spending: ${df_out['Amount'].sum():,.2f}",
            f"Avg Amount: ${df_out['Amount'].mean():,.2f}",
            f"Anomalies: {int(df_out['anomaly_flag'].sum()):,}",
        ] + insights

        pdf_bytes = insights_to_pdf_bytes("Financial Insight Generator Report", pdf_lines)

        st.download_button(
            label="Download PDF report",
            data=pdf_bytes,
            file_name="financial_insights_report.pdf",
            mime="application/pdf",
        )

except Exception as e:
    st.error(f"Error processing file: {e}")

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Financial Insight Generator", layout="wide")
st.title("Financial Insight Generator")
st.write("Upload a CSV of transactions to generate spending insights, anomalies, clustering, and an exportable report.")


# -----------------------------
# Helpers
# -----------------------------
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempts to map common column variants to expected names.
    Expected minimum: Amount
    Optional: Date, Category, Merchant/Description
    """
    col_map = {}

    # Normalize column names to make matching easier
    normalized = {c: c.strip().lower() for c in df.columns}

    # Map Amount column
    for c, cl in normalized.items():
        if cl in ["amount", "amt", "transaction_amount", "value", "price"]:
            col_map[c] = "Amount"
            break

    # Map Date column (optional)
    for c, cl in normalized.items():
        if cl in ["date", "transaction_date", "timestamp", "time"]:
            col_map[c] = "Date"
            break

    # Map Category column (optional)
    for c, cl in normalized.items():
        if cl in ["category", "type", "expense_category"]:
            col_map[c] = "Category"
            break

    # Map Merchant column (optional)
    for c, cl in normalized.items():
        if cl in ["merchant", "vendor", "payee"]:
            col_map[c] = "Merchant"
            break

    # If you have "description" but not merchant, treat as merchant-like
    if "Merchant" not in col_map.values():
        for c, cl in normalized.items():
            if cl in ["description", "details", "note", "narration"]:
                col_map[c] = "Merchant"
                break

    df = df.rename(columns=col_map)
    return df


def _parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Date if present and create time features."""
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # Basic time features
        df["month"] = df["Date"].dt.month
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["day"] = df["Date"].dt.day
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    else:
        # If no date, create empty columns to keep pipeline stable
        df["month"] = np.nan
        df["day_of_week"] = np.nan
        df["day"] = np.nan
        df["is_weekend"] = np.nan
    return df


def _winsorize_amount(df: pd.DataFrame, cap_percentile: float) -> pd.DataFrame:
    """
    Caps Amount to an upper percentile to make charts more interpretable.
    Keeps original Amount in Amount_raw.
    """
    df = df.copy()
    df["Amount_raw"] = df["Amount"].astype(float)

    cap_value = np.nanpercentile(df["Amount_raw"], cap_percentile)
    df["Amount"] = np.minimum(df["Amount_raw"], cap_value)

    return df


def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix for ML.
    Uses Amount + time features + category/merchant frequencies if available.
    """
    df = df.copy()

    # Frequency encodings help anomaly detection beyond raw amount
    if "Category" in df.columns:
        df["category_freq"] = df["Category"].astype(str).map(df["Category"].astype(str).value_counts())
    else:
        df["category_freq"] = 0

    if "Merchant" in df.columns:
        df["merchant_freq"] = df["Merchant"].astype(str).map(df["Merchant"].astype(str).value_counts())
    else:
        df["merchant_freq"] = 0

    # Fill NaNs
    for c in ["month", "day_of_week", "day", "is_weekend"]:
        df[c] = df[c].fillna(-1)

    feature_cols = ["Amount", "month", "day_of_week", "day", "is_weekend", "category_freq", "merchant_freq"]
    X = df[feature_cols].astype(float)

    return X


def analyze_transactions(df: pd.DataFrame, contamination: float, k: int) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """
    Runs IsolationForest for anomalies, KMeans for clusters,
    and RandomForestClassifier to predict anomaly label (explainability via feature importances).
    Returns:
      - insights list (strings)
      - enriched dataframe
      - feature importance dataframe
    """
    df = df.copy()

    # Feature matrix
    X = _build_feature_matrix(df)

    # Anomaly detection
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_pred = iso.fit_predict(X)  # -1 anomaly, 1 normal
    df["anomaly_label"] = np.where(iso_pred == -1, "Anomaly", "Normal")

    # Clustering (use Amount + simple time features for better separation)
    # Note: KMeans needs finite values; X is numeric and filled.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X[["Amount", "month", "day_of_week", "category_freq", "merchant_freq"]])

    # Build insights
    insights = []

    total_spent = df["Amount_raw"].sum() if "Amount_raw" in df.columns else df["Amount"].sum()
    avg_spent = df["Amount_raw"].mean() if "Amount_raw" in df.columns else df["Amount"].mean()
    anomaly_count = (df["anomaly_label"] == "Anomaly").sum()

    insights.append(f"Total Spending: ${total_spent:,.2f}")
    insights.append(f"Average Transaction: ${avg_spent:,.2f}")
    insights.append(f"Anomalies Detected: {anomaly_count} transactions flagged as unusual (IsolationForest).")

    # High spenders (use raw for accuracy)
    raw = df["Amount_raw"] if "Amount_raw" in df.columns else df["Amount"]
    threshold = raw.mean() + 2 * raw.std()
    high_spend = df[raw > threshold]
    if len(high_spend) > 0:
        insights.append(f"High-Value Outliers: {len(high_spend)} transactions exceed mean + 2*std.")

    # RandomForest explanation model
    # Train a simple classifier to approximate anomaly_label; feature importances give “why”
    y = (df["anomaly_label"] == "Anomaly").astype(int)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X, y)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    return insights, df, importance


def build_pdf_report(
    title: str,
    metrics: dict,
    insights: list[str],
    notes: str = ""
) -> bytes:
    """
    Generates a simple PDF report (text-based) for download.
    Uses fpdf2 (lightweight).
    """
    from fpdf import FPDF  # imported here so Streamlit runs without PDF dependency until needed

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, "Summary Metrics")
    pdf.set_font("Helvetica", "", 11)
    for k, v in metrics.items():
        pdf.multi_cell(0, 7, f"- {k}: {v}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, "Key Insights")
    pdf.set_font("Helvetica", "", 11)
    for item in insights:
        pdf.multi_cell(0, 7, f"- {item}")
    pdf.ln(2)

    if notes.strip():
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, "Notes")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, notes)

    return pdf.output(dest="S").encode("latin-1")


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")

cap_percentile = st.sidebar.slider(
    "Cap extreme values at percentile (for charts)",
    min_value=90,
    max_value=100,
    value=99,
    step=1,
    help="Caps large values to reduce skew and improve chart readability."
)

contamination = st.sidebar.slider(
    "IsolationForest contamination",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
    help="Expected fraction of anomalies in data."
)

k_clusters = st.sidebar.slider(
    "KMeans clusters",
    min_value=2,
    max_value=8,
    value=3,
    step=1
)

chart_choices = st.sidebar.multiselect(
    "Charts to display",
    [
        "Amount Histogram",
        "Amount Boxplot",
        "Clusters (Scatter)",
        "Spending Over Time",
        "Top Categories",
        "Top Merchants",
        "Anomalies Table",
        "Explainability (Feature Importance)"
    ],
    default=["Amount Histogram", "Amount Boxplot", "Clusters (Scatter)", "Top Categories", "Explainability (Feature Importance)"]
)

st.sidebar.markdown("---")
want_pdf = st.sidebar.checkbox("Enable PDF export", value=True)
notes_for_pdf = st.sidebar.text_area("Optional notes for PDF", "")


# -----------------------------
# Upload + Run
# -----------------------------
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = _standardize_columns(df)

        # Validate minimum columns
        if "Amount" not in df.columns:
            st.error("Your CSV must include an Amount column (or similar like 'amount', 'amt', 'value').")
            st.stop()

        # Parse date + create features
        df = _parse_date_column(df)

        # Cap outliers for charts (keep raw in Amount_raw)
        df = _winsorize_amount(df, cap_percentile=cap_percentile)

        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Run analysis button
        if st.button("Generate Insights", type="primary"):
            with st.spinner("Running ML analysis (anomalies + clustering + explanation)..."):
                insights, df_out, importance = analyze_transactions(
                    df=df,
                    contamination=contamination,
                    k=k_clusters
                )

            # Summary Metrics
            st.subheader("Summary Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{df_out.shape[0]}")
            col2.metric("Total Spending", f"${df_out['Amount_raw'].sum():,.2f}")
            col3.metric("Average Amount", f"${df_out['Amount_raw'].mean():,.2f}")
            col4.metric("Anomalies", f"{(df_out['anomaly_label']=='Anomaly').sum()}")

            st.subheader("Key Insights")
            for item in insights:
                st.write(f"- {item}")

            # -----------------------------
            # Plotly Charts (Interactive)
            # -----------------------------
            if "Amount Histogram" in chart_choices:
                st.subheader("Amount Distribution (Capped for readability)")
                fig = px.histogram(df_out, x="Amount", nbins=40, title="Histogram of Transaction Amounts (Capped)")
                st.plotly_chart(fig, use_container_width=True)

            if "Amount Boxplot" in chart_choices:
                st.subheader("Amount Spread (Boxplot - Capped)")
                fig = px.box(df_out, x="Amount", points="outliers", title="Boxplot of Transaction Amounts (Capped)")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            if "Clusters (Scatter)" in chart_choices:
                st.subheader("KMeans Clustering View")
                fig = px.scatter(
                    df_out.reset_index(),
                    x="index",
                    y="Amount",
                    color="cluster",
                    symbol="anomaly_label",
                    title="Clusters (color) with Anomalies (symbol) — Amount vs Index",
                    hover_data=[c for c in ["Date", "Category", "Merchant", "Amount_raw"] if c in df_out.columns]
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

            if "Spending Over Time" in chart_choices and "Date" in df_out.columns:
                st.subheader("Spending Over Time")
                df_ts = df_out.dropna(subset=["Date"]).copy()
                df_ts = df_ts.sort_values("Date")
                df_daily = df_ts.groupby(df_ts["Date"].dt.date)["Amount_raw"].sum().reset_index()
                df_daily.columns = ["Date", "DailySpend"]
                fig = px.line(df_daily, x="Date", y="DailySpend", title="Daily Spend (Raw Amount)")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

            if "Top Categories" in chart_choices and "Category" in df_out.columns:
                st.subheader("Top Categories by Spend")
                cat = df_out.groupby("Category")["Amount_raw"].sum().sort_values(ascending=False).head(10).reset_index()
                fig = px.bar(cat, x="Category", y="Amount_raw", title="Top 10 Categories by Total Spend")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

            if "Top Merchants" in chart_choices and "Merchant" in df_out.columns:
                st.subheader("Top Merchants by Spend")
                mer = df_out.groupby("Merchant")["Amount_raw"].sum().sort_values(ascending=False).head(10).reset_index()
                fig = px.bar(mer, x="Merchant", y="Amount_raw", title="Top 10 Merchants by Total Spend")
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

            if "Anomalies Table" in chart_choices:
                st.subheader("Flagged Anomalies")
                anomalies = df_out[df_out["anomaly_label"] == "Anomaly"].copy()
                show_cols = [c for c in ["Date", "Category", "Merchant", "Amount_raw", "Amount", "cluster"] if c in anomalies.columns]
                st.dataframe(anomalies[show_cols].sort_values("Amount_raw", ascending=False).head(50), use_container_width=True)

            if "Explainability (Feature Importance)" in chart_choices:
                st.subheader("What drove the anomaly decisions? (Explainability)")
                st.write(
                    "This uses a RandomForestClassifier trained to approximate the anomaly label. "
                    "Feature importances indicate which variables most influenced anomaly vs normal classification."
                )
                fig = px.bar(
                    importance.head(10),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top Feature Importances (Proxy Explainability)",
                )
                fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # PDF Export
            # -----------------------------
            if want_pdf:
                metrics = {
                    "Total Transactions": f"{df_out.shape[0]}",
                    "Total Spending": f"${df_out['Amount_raw'].sum():,.2f}",
                    "Average Transaction": f"${df_out['Amount_raw'].mean():,.2f}",
                    "Anomalies": f"{(df_out['anomaly_label']=='Anomaly').sum()}",
                    "Cap Percentile": f"{cap_percentile}th",
                    "IsolationForest contamination": f"{contamination}",
                    "KMeans clusters": f"{k_clusters}"
                }

                pdf_bytes = build_pdf_report(
                    title="Financial Insight Generator Report",
                    metrics=metrics,
                    insights=insights,
                    notes=notes_for_pdf
                )

                st.download_button(
                    label="Download Insights as PDF",
                    data=pdf_bytes,
                    file_name="financial_insights_report.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV to begin. Minimum requirement: a column named Amount (or similar).")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.inspection import permutation_importance

from fpdf import FPDF


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Financial Insight Generator", layout="wide")
st.title("Financial Insight Generator")
st.markdown("Upload a CSV of your transactions to receive automated insights, interactive charts, anomalies, clustering, and a PDF report.")

with st.expander("How to use this tool", expanded=True):
    st.markdown(
        """
**Minimum requirement**
- A column named `Amount` (or `amount`, `amt`, `value`)

**Optional (recommended)**
- `Date` (or `date`, `transaction_date`)
- `Category`
- `Merchant` or `Description`

**What you'll get**
- Anomaly detection (IsolationForest + optional LocalOutlierFactor)
- Spending clusters (KMeans spending tiers)
- Interactive charts (Plotly)
- PDF export of insights
"""
    )


# -----------------------------
# Helpers (CSV compatibility)
# -----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column names to canonical: Amount, Date, Category, Merchant."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c: c.lower().strip() for c in df.columns}

    def find(candidates):
        for c, cl in lower.items():
            if cl in candidates:
                return c
        return None

    rename = {}
    a = find(["amount", "amt", "value", "transaction_amount", "price"])
    d = find(["date", "transaction_date", "timestamp", "time", "posted_date"])
    cat = find(["category", "type", "expense_category"])
    mer = find(["merchant", "vendor", "payee", "merchant_name"])
    desc = find(["description", "details", "note", "narration"])

    if a: rename[a] = "Amount"
    if d: rename[d] = "Date"
    if cat: rename[cat] = "Category"
    if mer: rename[mer] = "Merchant"
    elif desc: rename[desc] = "Merchant"

    return df.rename(columns=rename)


def parse_amount(series: pd.Series) -> pd.Series:
    """Parse currency-like strings: $1,234.56 or (45.00)."""
    s = series.astype(str).str.strip()
    s = s.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    s = s.str.replace(r"[\$,]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["month"] = df["Date"].dt.month
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["day"] = df["Date"].dt.day
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    else:
        df["month"] = -1
        df["day_of_week"] = -1
        df["day"] = -1
        df["is_weekend"] = -1
    return df


def cap_amounts(series: pd.Series, q: float) -> pd.Series:
    cap = series.quantile(q)
    return np.minimum(series, cap)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric features for anomaly detection / explainability."""
    df = df.copy()

    if "Category" in df.columns:
        df["category_freq"] = df["Category"].astype(str).map(df["Category"].astype(str).value_counts())
    else:
        df["category_freq"] = 0

    if "Merchant" in df.columns:
        df["merchant_freq"] = df["Merchant"].astype(str).map(df["Merchant"].astype(str).value_counts())
    else:
        df["merchant_freq"] = 0

    cols = ["Amount_raw", "month", "day_of_week", "day", "is_weekend", "category_freq", "merchant_freq"]
    return df[cols].astype(float)


# -----------------------------
# PDF safety (fixes your error)
# -----------------------------
def _break_long_words(text: str, max_len: int = 40) -> str:
    out = []
    for token in str(text).split(" "):
        if len(token) > max_len:
            token = " ".join(token[i:i+max_len] for i in range(0, len(token), max_len))
        out.append(token)
    return " ".join(out)

def _pdf_safe(text: str) -> str:
    text = _break_long_words(text, max_len=40)
    return text.encode("latin-1", "replace").decode("latin-1")

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
# ML analysis (IsolationForest + optional LOF + KMeans + explainability proxy)
# -----------------------------
def analyze_transactions(df: pd.DataFrame, contamination: float, n_clusters: int, use_lof: bool):
    df = df.copy()

    X = build_features(df)

    # Anomaly model 1: IsolationForest
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_pred = iso.fit_predict(X)  # -1 anomaly
    df["anomaly_iforest"] = (iso_pred == -1).astype(int)

    # Anomaly model 2 (optional): LOF
    if use_lof and len(df) >= 50:
        n_neighbors = min(35, max(10, len(df)//20))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        lof_pred = lof.fit_predict(X)
        df["anomaly_lof"] = (lof_pred == -1).astype(int)
        df["anomaly_flag"] = ((df["anomaly_iforest"] == 1) | (df["anomaly_lof"] == 1)).astype(int)
    else:
        df["anomaly_lof"] = 0
        df["anomaly_flag"] = df["anomaly_iforest"]

    # Clustering: KMeans spending tiers on scaled amounts
    scaler = StandardScaler()
    amount_scaled = scaler.fit_transform(df[["Amount_raw"]].values)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(amount_scaled)

    # Insights
    total_spent = float(df["Amount_raw"].sum())
    avg_spent = float(df["Amount_raw"].mean())
    anomaly_count = int(df["anomaly_flag"].sum())
    anomaly_rate = float(df["anomaly_flag"].mean() * 100)

    high_spend = df[df["Amount_raw"] > df["Amount_raw"].mean() + 2 * df["Amount_raw"].std()]
    insights = [
        f"Total Spending: ${total_spent:,.2f}",
        f"Average Transaction: ${avg_spent:,.2f}",
        f"Anomalies Flagged: {anomaly_count:,} ({anomaly_rate:.2f}%)",
    ]
    if len(high_spend) > 0:
        insights.append(f"High-value outliers (> mean + 2×std): {len(high_spend):,}")

    # Explainability proxy: RF trained to mimic anomaly_flag + permutation importance
    # (Enticing “AI” without SHAP dependency)
    y = df["anomaly_flag"].astype(int)
    rf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=8, random_state=42)
    importance = pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean}).sort_values("importance", ascending=False)

    return insights, df, importance


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")

contamination = st.sidebar.slider("Anomaly sensitivity (contamination)", 0.01, 0.20, 0.05, 0.01)
n_clusters = st.sidebar.slider("KMeans clusters", 2, 8, 3, 1)

cap_toggle = st.sidebar.checkbox("Cap extreme values for charts", value=True)
cap_q = st.sidebar.slider("Cap quantile", 0.90, 0.999, 0.98, 0.001)

use_lof = st.sidebar.checkbox("Add 2nd anomaly model (LocalOutlierFactor)", value=True)

chart_choices = st.sidebar.multiselect(
    "Charts to display",
    [
        "Histogram",
        "Boxplot",
        "Cluster Scatter",
        "Spending Over Time",
        "Top Categories",
        "Top Merchants",
        "Anomalies Table",
        "Explainability (Feature Importance)",
    ],
    default=["Histogram", "Boxplot", "Cluster Scatter", "Anomalies Table", "Explainability (Feature Importance)"]
)

top_n = st.sidebar.slider("Top-N for categories/merchants", 5, 30, 10, 1)
want_pdf = st.sidebar.checkbox("Enable PDF export", value=True)
notes_for_pdf = st.sidebar.text_area("Optional notes for PDF", "")


# -----------------------------
# Upload UI
# -----------------------------
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin. Minimum requirement: `Amount` column (or amount/amt/value).")
    st.stop()


# -----------------------------
# Main workflow
# -----------------------------
try:
    df = pd.read_csv(uploaded_file)
    df = standardize_columns(df)

    if "Amount" not in df.columns:
        st.error("The uploaded file must contain an 'Amount' column (or similar like amount/amt/value).")
        st.stop()

    # Parse Amount
    df["Amount_raw"] = parse_amount(df["Amount"])
    df = df.dropna(subset=["Amount_raw"]).copy()

    if df.empty:
        st.error("No valid numeric Amount values were found.")
        st.stop()

    # Time features (optional)
    df = add_time_features(df)

    # For charts only (keep raw for metrics)
    df["Amount_plot"] = cap_amounts(df["Amount_raw"], cap_q) if cap_toggle else df["Amount_raw"]

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"**Total Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    if st.button("Generate Insights", type="primary"):
        with st.spinner("Analyzing your transactions..."):
            insights, df_out, importance = analyze_transactions(
                df=df,
                contamination=contamination,
                n_clusters=n_clusters,
                use_lof=use_lof,
            )

        # Summary Metrics
        st.subheader("Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{df_out.shape[0]:,}")
        col2.metric("Total Spending", f"${df_out['Amount_raw'].sum():,.2f}")
        col3.metric("Avg Transaction", f"${df_out['Amount_raw'].mean():,.2f}")
        col4.metric("Anomalies", f"{int(df_out['anomaly_flag'].sum()):,}")

        # Key Insights
        st.subheader("Key Insights")
        st.info("Here's what we found:")
        for insight in insights:
            st.markdown(f"- {insight}")

        # -----------------------------
        # Plotly Charts
        # -----------------------------
        st.subheader("Interactive Charts")

        if "Histogram" in chart_choices:
            fig = px.histogram(df_out, x="Amount_plot", nbins=40, title="Transaction Amounts (Capped if enabled)")
            st.plotly_chart(fig, use_container_width=True)

        if "Boxplot" in chart_choices:
            fig = px.box(df_out, x="Amount_plot", points="outliers", title="Amount Spread (Capped if enabled)")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        if "Cluster Scatter" in chart_choices:
            tmp = df_out.reset_index(drop=True)
            fig = px.scatter(
                tmp,
                x=tmp.index,
                y="Amount_plot",
                color="cluster",
                symbol=tmp["anomaly_flag"].map({0: "Normal", 1: "Anomaly"}),
                title="Spending Tiers (KMeans) + Anomaly Flags — Amount vs Index",
                hover_data=[c for c in ["Date", "Category", "Merchant", "Amount_raw"] if c in tmp.columns],
                opacity=0.75,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        if "Spending Over Time" in chart_choices:
            if "Date" in df_out.columns and df_out["Date"].notna().any():
                ts = df_out.dropna(subset=["Date"]).copy()
                ts = ts.sort_values("Date")
                daily = ts.groupby(ts["Date"].dt.date)["Amount_raw"].sum().reset_index()
                daily.columns = ["Date", "DailySpend"]
                fig = px.line(daily, x="Date", y="DailySpend", title="Daily Spend (Raw Amount)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Spending over time requires a Date column.")

        if "Top Categories" in chart_choices:
            if "Category" in df_out.columns:
                cat = df_out.groupby("Category")["Amount_raw"].sum().sort_values(ascending=False).head(top_n).reset_index()
                fig = px.bar(cat, x="Category", y="Amount_raw", title=f"Top {top_n} Categories by Spend")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Top Categories requires a Category column.")

        if "Top Merchants" in chart_choices:
            if "Merchant" in df_out.columns:
                mer = df_out.groupby("Merchant")["Amount_raw"].sum().sort_values(ascending=False).head(top_n).reset_index()
                fig = px.bar(mer, x="Merchant", y="Amount_raw", title=f"Top {top_n} Merchants by Spend")
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Top Merchants requires a Merchant/Description column.")

        if "Anomalies Table" in chart_choices:
            anomalies = df_out[df_out["anomaly_flag"] == 1].copy().sort_values("Amount_raw", ascending=False).head(50)
            if anomalies.empty:
                st.info("No anomalies flagged at this sensitivity.")
            else:
                cols = [c for c in ["Date", "Category", "Merchant", "Amount_raw", "cluster"] if c in anomalies.columns]
                st.dataframe(anomalies[cols], use_container_width=True)

        if "Explainability (Feature Importance)" in chart_choices:
            st.caption(
                "Explainability note: A RandomForest model is trained to mimic anomaly flags. "
                "Permutation importance shows which features most influence the proxy model."
            )
            fig = px.bar(
                importance.head(10),
                x="importance",
                y="feature",
                orientation="h",
                title="Top Feature Importances (Proxy Explainability)",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # PDF Export
        # -----------------------------
        if want_pdf:
            st.subheader("Download Report")

            pdf_lines = [
                f"Transactions: {len(df_out):,}",
                f"Total Spending: ${df_out['Amount_raw'].sum():,.2f}",
                f"Average Transaction: ${df_out['Amount_raw'].mean():,.2f}",
                f"Anomalies: {int(df_out['anomaly_flag'].sum()):,}",
            ] + insights

            if notes_for_pdf.strip():
                pdf_lines.append(f"Notes: {notes_for_pdf}")

            pdf_bytes = insights_to_pdf_bytes("Financial Insight Generator Report", pdf_lines)

            st.download_button(
                label="Download PDF report",
                data=pdf_bytes,
                file_name="financial_insights_report.pdf",
                mime="application/pdf",
            )

except Exception as e:
    st.error(f"Error processing file: {e}")

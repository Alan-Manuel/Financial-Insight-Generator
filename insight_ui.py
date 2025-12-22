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
    ins

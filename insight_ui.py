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
st.caption("Upload a CSV of transactions to generate interactive insights, anomaly detection, clustering, explainability, and a PDF report.")

with st.expander("How to use this tool (Read me first)", expanded=True):
    st.markdown(
        """
### What this does
- Flags unusual transactions using **IsolationForest** (and optional LocalOutlierFactor)
- Groups transactions into **spending tiers** using **KMeans**
- Provides interactive charts using **Plotly**
- Adds interpretability via an **explainability proxy** (RandomForest + permutation importance)
- Exports your summary to a **PDF report**

### CSV Requirements
Minimum:
- `Amount` column (numeric or currency-like strings)

Recommended:
- `Date` (parseable)
- `Category`
- `Merchant` or `Description`

If your columns are named differently, the app tries to map common aliases automatically.
        """
    )


# -----------------------------
# Helpers: make PDF safe
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
# Helpers: column mapping + coercion
# -----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    lower_map = {c: c.lower().strip() for c in df.columns}

    def find_col(candidates):
        for col, cl in lower_map.items():
            if cl in candidates:
                return col
        return None

    amount_col = find_col(["amount", "amt", "transaction_amount", "value", "price"])
    date_col = find_col(["date", "transaction_date", "timestamp", "time", "posted_date"])
    cat_col = find_col(["category", "type", "expense_category"])
    merch_col = find_col(["merchant", "vendor", "payee", "merchant_name"])
    desc_col = find_col(["description", "details", "note", "narration"])

    rename_map = {}
    if amount_col: rename_map[amount_col] = "Amount"
    if date_col: rename_map[date_col] = "Date"
    if cat_col: rename_map[cat_col] = "Category"
    if merch_col: rename_map[merch_col] = "Merchant"
    elif desc_col: rename_map[desc_col] = "Merchant"

    df = df.rename(columns=rename_map)
    return df


def parse_amount(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)  # (123) -> -123
    s = s.str.replace(r"[\$,]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["month"]

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
    df.columns = [str(c).strip() for c in]()

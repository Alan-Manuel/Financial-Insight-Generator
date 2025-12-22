import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import plotly.express as px
from fpdf import FPDF

# Optional LLM narrative (only used if you configure a key)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# App config + onboarding
# -----------------------------
st.set_page_config(page_title="Financial Insight Generator", layout="wide")
st.title("Financial Insight Generator")
st.caption("Upload a transactions CSV (or use demo data) to generate insights, anomaly detection, clustering, forecasting, and a PDF report.")

with st.expander("How to use this tool (Read me first)", expanded=True):
    st.markdown(
        """
### What this app does
- Cleans and validates your transaction data (so messy CSVs don’t crash the app)
- Detects unusual transactions with **IsolationForest** (anomaly detection)
- Groups transactions into spending tiers with **KMeans** (clustering)
- Shows interactive **Plotly** charts (zoom, hover, filter)
- Forecasts **future monthly spending** (simple baseline model)
- Exports a **PDF report** (metrics + insights + forecast summary)

### Required columns (case-insensitive)
- `date` (parseable date)
- `amount` (numeric; can include $, commas, parentheses)
- `category` (text)
- `merchant` (text)

### Tips
- If your CSV uses different names (e.g., `amt`), this app tries to auto-map common aliases.
- If category/merchant are missing, the app can optionally fill them as “Uncategorized/Unknown”.
        """
    )

# -----------------------------
# Schema expectations + alias mapping
# -----------------------------
REQUIRED_COLS = ["date", "amount", "category", "merchant"]

ALIASES = {
    "date": ["date", "transaction_date", "posted_date", "trans_date", "datetime", "timestamp"],
    "amount": ["amount", "amt", "value", "transaction_amount", "amount_usd", "amount($)"],
    "category": ["category", "cat", "type", "merchant_category", "mcc_category"],
    "merchant": ["merchant", "vendor", "payee", "description", "merchant_name", "name"],
    "account": ["account", "acct", "account_name"],
    # optional: if people upload "debit/credit" style
    "debit": ["debit", "debits", "withdrawal"],
    "credit": ["credit", "credits", "deposit"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to canonical names if possible."""
    df = df.copy()
    cols = set(df.columns)
    rename_map = {}

    for canonical, options in ALIASES.items():
        if canonical in cols:
            continue
        for opt in options:
            if opt in cols:
                rename_map[opt] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

def parse_amount_series(s: pd.Series) -> pd.Series:
    """
    Handles values like: "$1,234.56", "(45.10)", "1,200", etc.
    Returns numeric series with NaN for unparseable.
    """
    s = s.astype(str).str.strip()

    # Handle accounting negatives: (123.45) -> -123.45
    s = s.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)

    # Remove currency symbols and commas
    s = s.str.replace(r"[\$,]", "", regex=True)

    return pd.to_numeric(s, errors="coerce")

def validate_and_clean(df: pd.DataFrame, allow_fill_missing_text: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Returns cleaned df + a data quality report dict.
    """
    report = {
        "raw_rows": len(df),
        "raw_cols": list(df.columns),
        "dropped_rows": 0,
        "invalid_date_rows": 0,
        "invalid_amount_rows": 0,
        "missing_category_rows": 0,
        "missing_merchant_rows": 0,
        "final_rows": 0,
        "notes": [],
    }

    df = normalize_columns(df)
    df = apply_aliases(df)

    # If "amount" doesn't exist but debit/credit do, derive amount = debit - credit
    if "amount" not in df.columns and {"debit", "credit"}.issubset(df.columns):
        report["notes"].append("Derived amount from debit/credit: amount = debit - credit.")
        debit = parse_amount_series(df["debit"]).fillna(0)
        credit = parse_amount_series(df["credit"]).fillna(0)
        df["amount"] = debit - credit

    # Optionally fill missing text columns
    if allow_fill_missing_text:
        if "category" not in df.columns:
            df["category"] = "Uncategorized"
            report["notes"].append("Missing category column; filled with 'Uncategorized'.")
        if "merchant" not in df.columns:
            df["merchant"] = "Unknown"
            report["notes"].append("Missing merchant column; filled with 'Unknown'.")

    # Validate required cols
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        report["notes"].append(f"Missing required columns: {missing}")
        # return empty to force a friendly message upstream
        return pd.DataFrame(), report

    # Parse types (track invalids before dropping)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = parse_amount_series(df["amount"])

    report["invalid_date_rows"] = int(df["date"].isna().sum())
    report["invalid_amount_rows"] = int(df["amount"].isna().sum())
    report["missing_category_rows"] = int(df["category"].isna().sum())
    report["missing_merchant_rows"] = int(df["merchant"].isna().sum())

    # Drop bad rows
    before = len(df)
    df = df.dropna(subset=["date", "amount", "category", "merchant"])
    after = len(df)

    report["dropped_rows"] = int(before - after)
    report["final_rows"] = int(after)

    return df, report


# -----------------------------
# ML: anomalies + clusters
# -----------------------------
def analyze_transactions(df: pd.DataFrame, contamination: float, n_clusters: int) -> pd.DataFrame:
    df = df.copy()

    X = df[["amount"]].values

    iso = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly_flag"] = (iso.fit_predict(X) == -1).astype(int)  # 1 = anomaly

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    df["cluster"] = kmeans.fit_predict(X_scaled)

    return df

def cap_amounts(series: pd.Series, cap_quantile: float) -> pd.Series:
    cap_value = series.quantile(cap_quantile)
    return np.minimum(series, cap_value)


# -----------------------------
# Forecast: monthly spend baseline (trend + seasonality)
# -----------------------------
def forecast_monthly_spend(df: pd.DataFrame, horizon_months: int = 6) -> pd.DataFrame:
    """
    Forecast monthly spend using LinearRegression on:
      - time index (trend)
      - month-of-year one-hot (seasonality)
    Excludes refunds (amount <= 0) from spend forecast.
    """
    d = df.copy()
    d = d[d["amount"] > 0].copy()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    monthly = d.groupby("month")["amount"].sum().reset_index().sort_values("month")

    if len(monthly) < 4:
        # not enough history -> flat forecast
        last = float(monthly["amount"].iloc[-1]) if len(monthly) else 0.0
        future_months = pd.date_range(
            start=(monthly["month"].max() + pd.offsets.MonthBegin(1)) if len(monthly) else pd.Timestamp.today().replace(day=1),
            periods=horizon_months,
            freq="MS",
        )
        return pd.DataFrame({
            "month": list(monthly["month"]) + list(future_months),
            "actual": list(monthly["amount"]) + [np.nan] * horizon_months,
            "forecast": [np.nan] * len(monthly) + [last] * horizon_months,
        })

    monthly["t"] = np.arange(len(monthly))
    monthly["moy"] = monthly["month"].dt.month

    X = pd.get_dummies(monthly[["t", "moy"]], columns=["moy"], drop_first=True)
    y = monthly["amount"].values

    model = LinearRegression()
    model.fit(X, y)

    last_month = monthly["month"].max()
    future_months = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")
    future = pd.DataFrame({"month": future_months})
    future["t"] = np.arange(len(monthly), len(monthly) + horizon_months)
    future["moy"] = future["month"].dt.month

    Xf = pd.get_dummies(future[["t", "moy"]], columns=["moy"], drop_first=True)
    Xf = Xf.reindex(columns=X.columns, fill_value=0)

    yhat = model.predict(Xf)
    yhat = np.maximum(yhat, 0)

    return pd.DataFrame({
        "month": list(monthly["month"]) + list(future_months),
        "actual": list(monthly["amount"]) + [np.nan] * horizon_months,
        "forecast": [np.nan] * len(monthly) + list(yhat),
    })


# -----------------------------
# Narrative: rule-based insights + optional LLM
# -----------------------------
def rule_based_insights(df: pd.DataFrame) -> list[str]:
    total_spend = float(df["amount"].sum())
    avg_txn = float(df["amount"].mean())
    anomaly_rate = float(df["anomaly_flag"].mean() * 100)

    top_category = df.groupby("category")["amount"].sum().sort_values(ascending=False).head(1).index[0]
    top_merchant = df.groupby("merchant")["amount"].sum().sort_values(ascending=False).head(1).index[0]

    high_spend = df[df["amount"] > df["amount"].mean() + 2 * df["amount"].std()]
    n_high = len(high_spend)

    bullets = [
        f"Total spend is **${total_spend:,.2f}** across **{len(df):,}** transactions (avg **${avg_txn:,.2f}**).",
        f"Top category is **{top_category}**; top merchant by total spend is **{top_merchant}**.",
        f"Anomaly detection flagged **{anomaly_rate:.2f}%** of transactions as unusual.",
    ]

    if n_high > 0:
        bullets.append(f"Detected **{n_high:,}** high-value transactions (> mean + 2×std). Consider adding review/alerts.")

    if anomaly_rate >= 6:
        bullets.append("Recommendation: add a stricter review rule for top 1–2% largest transactions or anomalies.")
    else:
        bullets.append("Recommendation: do a weekly review of the top 20 transactions + anomalies flagged.")

    bullets.append("Recommendation: look for recurring costs in top merchants/categories and consider consolidating or renegotiating.")
    return bullets

def generate_llm_summary(stats: dict) -> str:
    if OpenAI is None:
        return "LLM library not installed. Add `openai` to requirements.txt to enable."
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return "No LLM configured. Add OPENAI_API_KEY to Streamlit secrets to enable narrative insights."

    client = OpenAI(api_key=api_key)
    prompt = f"""
You are a personal finance analytics assistant. Write a concise, practical insight summary.
Use these computed metrics (do not invent numbers):
{stats}

Include:
- 2–3 key observations
- 2 action-oriented recommendations
- mention any notable anomaly behavior
Keep it under 120 words.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# PDF export
# -----------------------------
def insights_to_pdf_bytes(title: str, lines: list[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)

    # Helvetica is safer than Arial in many environments
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.multi_cell(0, 8, title)
    pdf.ln(2)

    pdf.set_font("Helvetica", size=11)
    for line in lines:
        pdf.multi_cell(0, 6, f"- {line}")

    return pdf.output(dest="S").encode("latin-1")


# -----------------------------
# UI controls
# -----------------------------
st.sidebar.header("Data source")
use_demo = st.sidebar.checkbox("Use demo dataset included in repo", value=True)

st.sidebar.header("Model settings")
contamination = st.sidebar.slider("Anomaly sensitivity (IsolationForest contamination)", 0.01, 0.15, 0.05, 0.01)
n_clusters = st.sidebar.slider("Number of clusters (KMeans)", 2, 8, 3, 1)

st.sidebar.header("Chart settings")
cap_toggle = st.sidebar.checkbox("Cap extreme values for charts", value=True)
cap_q = st.sidebar.slider("Cap quantile (higher = less capping)", 0.90, 0.999, 0.98, 0.001)

chart_set = st.sidebar.multiselect(
    "Charts to show",
    [
        "Daily Spending Trend",
        "Monthly Spend Trend",
        "Category Spend",
        "Category Share (Donut)",
        "Monthly Heatmap",
        "Anomaly Rate (Weekly)",
        "Top Merchants",
        "Recurring Merchants",
        "Anomaly Transactions Table",
        "Box Plot",
        "Cluster Scatter",
        "Future Spend Forecast (Monthly)",
    ],
    default=["Monthly Spend Trend", "Category Spend", "Anomaly Rate (Weekly)", "Future Spend Forecast (Monthly)"]
)

top_n_merchants = st.sidebar.slider("Top merchants", 5, 30, 10, 1)

st.sidebar.header("Forecast")
forecast_months = st.sidebar.slider("Forecast horizon (months)", 3, 12, 6, 1)

st.sidebar.header("Narrative")
show_llm = st.sidebar.checkbox("Generate optional LLM narrative (requires API key)", value=False)
show_raw = st.sidebar.checkbox("Show processed dataset preview", value=False)

st.sidebar.header("Validation")
allow_fill_missing_text = st.sidebar.checkbox("Fill missing category/merchant if absent", value=True)


# -----------------------------
# Load data
# -----------------------------
df_raw = None
demo_path = os.path.join("data", "fake_transactions_10k.csv")

if use_demo:
    if not os.path.exists(demo_path):
        st.error(f"Demo dataset not found at `{demo_path}`. Add your CSV there or turn off demo mode.")
        st.stop()
    df_raw = pd.read_csv(demo_path)
    st.success("Loaded demo dataset: data/fake_transactions_10k.csv")
else:
    uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to begin, or enable the demo dataset from the sidebar.")
        st.stop()
    df_raw = pd.read_csv(uploaded_file)


# -----------------------------
# Validate + clean
# -----------------------------
df_clean, report = validate_and_clean(df_raw, allow_fill_missing_text=allow_fill_missing_text)

st.subheader("Data Quality Check")
st.write("Detected columns:", report["raw_cols"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows (raw)", report["raw_rows"])
c2.metric("Invalid dates", report["invalid_date_rows"])
c3.metric("Invalid amounts", report["invalid_amount_rows"])
c4.metric("Dropped rows", report["dropped_rows"])
c5.metric("Rows (clean)", report["final_rows"])

if report["notes"]:
    st.info("Notes:\n- " + "\n- ".join(report["notes"]))

if df_clean.empty:
    st.error(
        "Could not proceed because required columns were missing or too many rows were invalid.\n\n"
        "Required columns are: date, amount, category, merchant (case-insensitive)."
    )
    st.stop()

if len(df_clean) < 50:
    st.warning("Very small dataset after cleaning. Charts/models may be unstable. Try a dataset with more rows.")


# -----------------------------
# Run analysis
# -----------------------------
df_out = analyze_transactions(df_clean, contamination=contamination, n_clusters=n_clusters)

# Capped copy for charts only
df_plot = df_out.copy()
df_plot["amount_plot"] = cap_amounts(df_plot["amount"], cap_q) if cap_toggle else df_plot["amount"]

# Summary metrics
total_spend = float(df_out["amount"].sum())
avg_txn = float(df_out["amount"].mean())
anomaly_rate = float(df_out["anomaly_flag"].mean() * 100)
top_category = df_out.groupby("category")["amount"].sum().sort_values(ascending=False).head(1).index[0]

st.subheader("Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions", f"{len(df_out):,}")
m2.metric("Total spend", f"${total_spend:,.2f}")
m3.metric("Avg transaction", f"${avg_txn:,.2f}")
m4.metric("Anomaly rate", f"{anomaly_rate:.2f}%")


# -----------------------------
# Insights
# -----------------------------
st.subheader("Key Insights")
bullets = rule_based_insights(df_out)
for b in bullets:
    st.markdown(f"- {b}")

if show_llm:
    st.subheader("Narrative Summary (Optional LLM)")
    stats_for_llm = {
        "transactions": len(df_out),
        "total_spend": round(total_spend, 2),
        "avg_transaction": round(avg_txn, 2),
        "anomaly_rate_pct": round(anomaly_rate, 2),
        "top_category": top_category,
    }
    st.write(generate_llm_summary(stats_for_llm))


# -----------------------------
# Charts
# -----------------------------
st.subheader("Interactive Charts")

if not chart_set:
    st.info("No charts selected. Use the sidebar to add charts.")
else:
    tabs = st.tabs(chart_set)
    for tab_name, tab in zip(chart_set, tabs):
        with tab:
            if tab_name == "Daily Spending Trend":
                daily = df_plot.groupby(pd.Grouper(key="date", freq="D"))["amount_plot"].sum().reset_index()
                fig = px.line(daily, x="date", y="amount_plot", title="Daily Spending Trend")
                st.plotly_chart(fig, use_container_width=True)

            elif tab_name == "Monthly Spend Trend":
                monthly = df_plot.copy()
                monthly["month"] = monthly["date"].dt.to_period("M").dt.to_timestamp()
                monthly = monthly.groupby("month")["amount_plot"].sum().reset_index()
                fig = px.line(monthly, x="month", y="amount_plot", title="Monthly Spend Trend")
                st.plotly_chart(fig, use_container_width=True)

            elif tab_name == "Category Spend":
                cat_spend = df_plot.groupby("category")["amount_plot"].sum().sort_values(ascending=False).reset_index()
                fig = px.bar(cat_spend, x="category", y="amount_plot", title="Total Spend by Category")
                st.plotly_chart(fig, use_container_width=True)

            elif tab_name == "Category Share (Donut)":
                cat = df_plot.groupby("category")["amount_plot"].sum().sort_values(ascending=False).reset_index()
                fig = px.pie(cat, names="category", values="amount_plot", hole=0.45, title="Category Share of Spend")
                st.plotly_chart(fig, use_container_width=True)

            elif tab_name == "Monthly Heatmap":
                tmp = df_plot.copy()
                tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
                heat = tmp.pivot_table(index="category", columns="month", values="amount_plot", aggfunc="sum", fill_value=0)
                heat_reset = heat.reset_index().melt(id_vars="category", var_name="month", value_name="spend")
                fig = px.density_heatmap(
                    heat_reset, x="month", y="category", z="spend",
                    title="Monthly Spending Heatmap (Category vs Month)",
                )
                st.plotly_chart(fig, use_container_width=True)

            elif tab_name == "Anomaly Rate (Weekly)":
                weekly = df_out.groupby(pd.Grouper(key="date", freq="W"))["anomaly_flag"].mean().reset_index()
                fig = px.line(weekly, x="date", y="anomaly_flag", title="Anomaly Rate Over Time (Weekly)")
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

            elif tab_name == "Top Merchants":
                merch = (
                    df_plot.groupby("merchant")["amount_plot"].sum()
                    .sort_values(ascending=False).head(top_n_merchants).reset_index()
                )
                fig = px.bar(merch, x="merchant", y="amount_plot", title=f"Top {top_n_merchants} Merchants by Spend")
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

            elif tab_name == "Recurring Merchants":
                tmp = df_out[df_out["amount"] > 0].copy()
                tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
                merchant_months = tmp.groupby("merchant")["month"].nunique().sort_values(ascending=False)
                recurring = merchant_months[merchant_months >= 3].head(25)

                if recurring.empty:
                    st.info("No recurring merchants detected using heuristic: appearing in ≥ 3 different months.")
                else:
                    rec_df = recurring.reset_index()
                    rec_df.columns = ["merchant", "months_active"]
                    fig = px.bar(rec_df, x="merchant", y="months_active", title="Recurring Merchants (≥ 3 months active)")
                    fig.update_layout(xaxis_tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Heuristic: merchants appearing in 3+ distinct months may be subscriptions/recurring bills.")

            elif tab_name == "Anomaly Transactions Table":
                anomalies = df_out[df_out["anomaly_flag"] == 1].copy().sort_values("amount", ascending=False).head(50)
                if anomalies.empty:
                    st.info("No anomalies flagged at the current sensitivity.")
                else:
                    st.dataframe(anomalies[["date", "merchant", "category", "amount", "cluster"]], use_container_width=True)

            elif tab_name == "Box Plot":
                fig = px.box(df_plot, y="amount_plot", points="outliers", title="Transaction Value Spread (Box Plot)")
                st.plotly_chart(fig, use_container_width=True)

            elif tab_name == "Cluster Scatter":
                tmp = df_plot.reset_index(drop=True)
                fig = px.scatter(
                    tmp,
                    x=tmp.index,
                    y="amount_plot",
                    color="cluster",
                    title="KMeans Clustering (Index vs Amount)",
                    opacity=0.7,
                )
                st.plotly_chart(fig, use_container_width=True)

            elif tab_name == "Future Spend Forecast (Monthly)":
                fc = forecast_monthly_spend(df_out, horizon_months=forecast_months)
                fc["series"] = fc["actual"].combine_first(fc["forecast"])
                fig = px.line(fc, x="month", y="series", title="Monthly Spend: Actual + Forecast")
                st.plotly_chart(fig, use_container_width=True)

                future_only = fc[fc["forecast"].notna()][["month", "forecast"]].copy()
                future_only["month"] = future_only["month"].dt.strftime("%Y-%m")
                future_only["forecast"] = future_only["forecast"].round(2)
                st.dataframe(future_only, use_container_width=True)


# -----------------------------
# Download PDF report
# -----------------------------
st.subheader("Download Report")

# Forecast summary for PDF
forecast_lines = []
fc = forecast_monthly_spend(df_out, horizon_months=forecast_months)
future_only = fc[fc["forecast"].notna()].copy()
if not future_only.empty:
    next_month_forecast = float(future_only.iloc[0]["forecast"])
    horizon_total = float(future_only["forecast"].sum())
    forecast_lines = [
        f"Forecast horizon: {forecast_months} months",
        f"Next month forecast (spend): ${next_month_forecast:,.2f}",
        f"Forecast total over horizon: ${horizon_total:,.2f}",
        "Forecast note: baseline model using historical monthly trend + seasonality; accuracy may vary.",
    ]

pdf_lines = [
    f"Transactions (clean): {len(df_out):,}",
    f"Total spend: ${total_spend:,.2f}",
    f"Avg transaction: ${avg_txn:,.2f}",
    f"Anomaly rate: {anomaly_rate:.2f}%",
    f"Top category: {top_category}",
] + forecast_lines + bullets

pdf_bytes = insights_to_pdf_bytes("Financial Insight Generator Report", pdf_lines)

st.download_button(
    label="Download PDF report",
    data=pdf_bytes,
    file_name="financial_insights_report.pdf",
    mime="application/pdf",
)

if show_raw:
    with st.expander("Processed Data (Preview)"):
        st.dataframe(df_out.head(200), use_container_width=True)

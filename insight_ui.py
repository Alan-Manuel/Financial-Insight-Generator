# insight_ui.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Streamlit Page Setup
st.set_page_config(page_title="💸 Financial Insight Generator", layout="centered")
st.title("💸 Financial Insight Generator")
st.markdown("Upload a CSV of your transactions to receive automated insights.")

# Function to analyze transaction data
def analyze_transactions(df: pd.DataFrame):
    result = []

    # Run Isolation Forest for anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_score"] = iso.fit_predict(df[["Amount"]])

    # Run KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["Amount"]])

    # Flag high-value outliers
    high_spend = df[df["Amount"] > df["Amount"].mean() + 2 * df["Amount"].std()]
    if not high_spend.empty:
        result.append(f"⚠️ Detected {len(high_spend)} unusually high-value transactions.")

    # Summary metrics
    total_spent = df["Amount"].sum()
    avg_spent = df["Amount"].mean()
    result.append(f"💰 Total Spending: ${total_spent:.2f}")
    result.append(f"📊 Average Transaction: ${avg_spent:.2f}")

    # Optional: Placeholder LLM-style insight
    result.append("🧠 Insight: Consider setting alerts for outlier spending to manage finances better.")

    return result

# Upload UI
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Basic validation
        if "Amount" not in df.columns:
            st.error("❌ The uploaded file must contain an 'Amount' column.")
        else:
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())

            # Trigger analysis
            if st.button("Generate Insights"):
                with st.spinner("Analyzing your transactions..."):
                    insights = analyze_transactions(df)
                    st.success("Insights Generated:")
                    for insight in insights:
                        st.write(insight)
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Streamlit Page Setup
st.set_page_config(page_title="💸 Financial Insight Generator", layout="centered")
st.title("💸 Financial Insight Generator")
st.markdown("Upload a CSV of your transactions to receive automated insights and visual analytics.")

# Function to analyze transaction data
def analyze_transactions(df: pd.DataFrame):
    result = []

    # Isolation Forest for anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_score"] = iso.fit_predict(df[["Amount"]])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["Amount"]])

    # Detect high spenders
    high_spend = df[df["Amount"] > df["Amount"].mean() + 2 * df["Amount"].std()]
    if not high_spend.empty:
        result.append(f"⚠️ **{len(high_spend)} unusually high-value transactions** detected.")

    # Summary
    total_spent = df["Amount"].sum()
    avg_spent = df["Amount"].mean()
    result.append(f"💰 **Total Spending:** ${total_spent:,.2f}")
    result.append(f"📊 **Average Transaction:** ${avg_spent:,.2f}")
    result.append("🧠 _Insight_: Consider setting alerts for outlier spending to manage finances better.")

    return result, df

# Upload UI
uploaded_file = st.file_uploader("📄 Upload Transaction CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "Amount" not in df.columns:
            st.error("❌ The uploaded file must contain an 'Amount' column.")
        else:
            st.subheader("📋 Preview of Uploaded Data")
            st.dataframe(df.head())

            st.write(f"✅ **Total Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

            if st.button("🔍 Generate Insights"):
                with st.spinner("Analyzing your transactions..."):
                    insights, df = analyze_transactions(df)

                    # Summary Metrics
                    st.subheader("📈 Summary Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Transactions", f"{df.shape[0]}")
                    col2.metric("Total Spending", f"${df['Amount'].sum():,.2f}")
                    col3.metric("Avg Transaction", f"${df['Amount'].mean():,.2f}")

                    # Chart 1: Histogram
                    st.subheader("📊 Distribution of Transaction Amounts")
                    fig1, ax1 = plt.subplots()
                    ax1.hist(df["Amount"], bins=30, color="skyblue", edgecolor="black")
                    ax1.set_xlabel("Transaction Amount")
                    ax1.set_ylabel("Frequency")
                    st.pyplot(fig1)

                    # Chart 2: Boxplot
                    st.subheader("📦 Transaction Value Spread (Boxplot)")
                    fig2, ax2 = plt.subplots()
                    ax2.boxplot(df["Amount"], vert=False)
                    ax2.set_xlabel("Amount")
                    st.pyplot(fig2)

                    # Chart 3: Cluster Scatter Plot
                    st.subheader("🎯 Transaction Clustering via KMeans")
                    fig3, ax3 = plt.subplots()
                    colors = {0: "blue", 1: "green", 2: "red"}
                    for cluster in df["cluster"].unique():
                        cluster_data = df[df["cluster"] == cluster]
                        ax3.scatter(cluster_data.index, cluster_data["Amount"],
                                    label=f"Cluster {cluster}", color=colors.get(cluster, "gray"))
                    ax3.set_xlabel("Transaction Index")
                    ax3.set_ylabel("Amount")
                    ax3.legend()
                    st.pyplot(fig3)

                    # Insights
                    st.subheader("🧠 Key Insights")
                    st.info("Here's what we found from your data:")
                    for insight in insights:
                        st.markdown(f"- {insight}")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

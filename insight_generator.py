# insight_generator.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Dummy insight generation using basic logic and placeholder LLM
def analyze_transactions(df: pd.DataFrame):
    result = []

    # Run Isolation Forest for anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_score"] = iso.fit_predict(df[["Amount"]])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["Amount"]])

    # Example: flag high-value outliers
    high_spend = df[df["Amount"] > df["Amount"].mean() + 2 * df["Amount"].std()]
    if not high_spend.empty:
        result.append(f"⚠️ Detected {len(high_spend)} unusually high-value transactions.")

    # Summary
    total_spent = df["Amount"].sum()
    avg_spent = df["Amount"].mean()
    result.append(f"💰 Total Spending: ${total_spent:.2f}")
    result.append(f"📊 Average Transaction: ${avg_spent:.2f}")
    
    # Placeholder: Add OpenAI/GPT narrative here if needed
    result.append("🧠 Insight: Consider setting alerts for outlier spending to manage finances better.")

    return result

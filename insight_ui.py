# insight_ui.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="💸 Financial Insight Generator", layout="centered")
st.title("💸 Financial Insight Generator")
st.markdown("Upload a CSV of your transactions to receive automated insights.")

# Upload UI
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Send file to FastAPI
    if st.button("Generate Insights"):
        with st.spinner("Analyzing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post("http://localhost:8000/analyze/", files=files)

            if response.status_code == 200:
                insights = response.json()["insights"]
                st.success("Insights Generated:")
                for insight in insights:
                    st.write(insight)
            else:
                st.error("Something went wrong. Try again.")

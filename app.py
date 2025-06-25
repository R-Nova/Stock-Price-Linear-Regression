import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
st.sidebar.markdown("---")
st.sidebar.info("Make sure your file has a **'Close'** column.")

# ---------- MAIN TITLE ----------
st.markdown("<h1 style='text-align: center;'>📈 Stock Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

# ---------- DATA PROCESSING ----------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Close' not in df.columns:
        st.warning("Your file must have a 'Close' column.")
    else:
        df['PrevClose'] = df['Close'].shift(1)
        df.dropna(inplace=True)

        X = df[['PrevClose']].values
        y = df['Close'].values

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # ---------- METRICS ----------
        latest_price = df['Close'].iloc[-1]
        predicted_price = predictions[-1]

        col1, col2 = st.columns(2)
        col1.metric("📌 Latest Closing Price", f"${latest_price:.2f}")
        col2.metric("🔮 Model Prediction", f"${predicted_price:.2f}")

        # ---------- CHART ----------
        st.write("---")
        st.subheader("📉 Actual vs Predicted Prices")

        plt.style.use("seaborn-darkgrid")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Close'].values, label='Actual Price', color='skyblue', linewidth=2)
        ax.plot(predictions, label='Predicted Price', color='orange', linestyle='--')
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.set_title("Stock Price Prediction")
        ax.legend()
        st.pyplot(fig)

        st.success("✅ Prediction complete! Upload a different file to try again.")

else:
    st.info("📂 Please upload a CSV file to begin.")

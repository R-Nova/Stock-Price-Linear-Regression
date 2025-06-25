import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf
from io import BytesIO

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("📡 Live Stock Data")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. GOOG, AAPL)", "GOOG")
days = st.sidebar.slider("Number of Past Days", 60, 365, 180)

# ---------- TITLE ----------
st.markdown("<h1 style='text-align: center;'>📈 Stock Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

# ---------- FETCH DATA ----------
try:
    df = yf.download(ticker, period=f"{days}d")
    df = df[['Close']]
    df.dropna(inplace=True)
    st.subheader(f"Showing Data for: {ticker}")
    st.write(df.tail())

    # ---------- PREDICTION ----------
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
    col1.metric("📌 Latest Close", f"${latest_price:.2f}")
    col2.metric("🔮 Model Prediction", f"${predicted_price:.2f}")

    # ---------- CHART ----------
    st.write("---")
    st.subheader("📉 Actual vs Predicted Prices")
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'].values, label='Actual', color='skyblue', linewidth=2)
    ax.plot(predictions, label='Predicted', color='orange', linestyle='--')
    ax.set_title("Stock Price Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # ---------- DOWNLOAD BUTTON ----------
    st.write("---")
    result_df = df.copy()
    result_df['Predicted_Close'] = predictions
    csv = result_df.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name=f'{ticker}_predictions.csv',
        mime='text/csv',
    )

    st.success("✅ Done! You can try another stock from the sidebar.")

except Exception as e:
    st.error(f"⚠️ Error loading stock data: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf
from io import BytesIO

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Stock Predictor", page_icon="??", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("?? Live Stock Data")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. GOOG, AAPL)", "GOOG")
days = st.sidebar.slider("Number of Past Days", 60, 365, 180)

# ---------- TITLE ----------
st.markdown("<h1 style='text-align: center;'>?? Stock Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

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

    # Shift predictions by one step to simulate "next-day" forecast
    shifted_predictions = np.append([np.nan], predictions[:-1])

    # Predict tomorrow’s price using the latest known price
    latest_close = df['Close'].iloc[-1]
    tomorrow_prediction = model.predict(np.array([[latest_close]]))[0]

    # ---------- METRICS ----------
    col1, col2, col3 = st.columns(3)
    col1.metric("?? Latest Close", f"${latest_close:.2f}")
    col2.metric("?? Model Prediction", f"${predictions[-1]:.2f}")
    col3.metric("?? Tomorrow's Prediction", f"${tomorrow_prediction:.2f}")

    # ---------- CHART ----------
    st.write("---")
    st.subheader("?? Actual vs Predicted Prices")

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'].values, label='Actual', color='skyblue', linewidth=2)
    ax.plot(shifted_predictions, label='Predicted (Next Day)', color='orange', linestyle='--', linewidth=2)
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()

    # Show chart in Streamlit
    st.pyplot(fig)

    # ---------- DOWNLOAD CHART AS JPEG ----------
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='jpeg')
    st.download_button(
        label="?? Download Chart as JPEG",
        data=img_buffer.getvalue(),
        file_name=f"{ticker}_chart.jpg",
        mime="image/jpeg"
    )

    # ---------- DOWNLOAD PREDICTIONS CSV ----------
    st.write("---")
    result_df = df.copy()
    result_df['Predicted_Close'] = shifted_predictions
    csv = result_df.to_csv(index=False).encode()
    st.download_button(
        label="?? Download Predictions as CSV",
        data=csv,
        file_name=f'{ticker}_predictions.csv',
        mime='text/csv',
    )

    st.success("? Prediction complete! Try another stock from the sidebar.")

except Exception as e:
    st.error(f"?? Error loading stock data: {e}")
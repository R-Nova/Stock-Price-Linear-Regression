import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf
from io import BytesIO

st.set_page_config(page_title="MUHAF Multi-Asset Predictor", page_icon="📊", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.title("📊 Choose Data Source")
data_mode = st.sidebar.radio("How do you want to input data?", ["📂 Upload CSV", "🌐 Fetch from Internet"])

# ---------- SETTINGS ----------
days = st.sidebar.slider("Number of Past Days (for live fetch)", 60, 365, 180)
symbols_input = st.sidebar.text_input("Enter symbols (for fetch mode only)", "AAPL,BTC-USD")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (must include 'Close' column)", type=["csv"])

# ---------- MAIN TITLE ----------
st.title("📈 MUHAF")
st.write("Welcome to the MUHAF Stock Price Prediction Dashboard!")

results = {}
fig, ax = plt.subplots(figsize=(14, 7))
colors = plt.cm.tab10.colors

# ---------- PROCESS CSV UPLOAD ----------
if data_mode == "📂 Upload CSV":
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = df[['Close']].dropna()
            df['Prev'] = df['Close'].shift(1)
            df.dropna(inplace=True)

            X = df[['Prev']].values
            y = df['Close'].values

            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(X)

            actual = df['Close'].values
            latest = float(actual[-1])
            predicted_today = float(preds[-1])
            tomorrow = model.predict(np.array([[latest]]))[0]

            ax.plot(actual, label='Actual (CSV)', color='skyblue', linewidth=2)
            ax.plot(preds, label='Predicted (CSV)', color='orange', linestyle='--', linewidth=2)

            results['CSV'] = {
                'df': df,
                'preds': preds,
                'latest': latest,
                'predicted_today': predicted_today,
                'tomorrow': tomorrow
            }
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")
    else:
        st.info("📁 Please upload a CSV file.")

# ---------- PROCESS SYMBOLS ----------
elif data_mode == "🌐 Fetch from Internet":
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    if not symbols:
        st.warning("Please enter at least one symbol.")
    else:
        for i, sym in enumerate(symbols):
            try:
                df = yf.download(sym, period=f"{days}d")
                df = df[['Close']].dropna()
                df['Prev'] = df['Close'].shift(1)
                df.dropna(inplace=True)

                X = df[['Prev']].values
                y = df['Close'].values

                model = LinearRegression()
                model.fit(X, y)
                preds = model.predict(X)

                actual = df['Close'].values
                latest = float(actual[-1])
                predicted_today = float(preds[-1])
                tomorrow = model.predict(np.array([[latest]]))[0]

                ax.plot(actual, label=f"{sym} Actual", color=colors[i % len(colors)], linestyle='-')
                ax.plot(preds, label=f"{sym} Predicted", color=colors[i % len(colors)], linestyle='--')

                results[sym] = {
                    'df': df,
                    'preds': preds,
                    'latest': latest,
                    'predicted_today': predicted_today,
                    'tomorrow': tomorrow
                }
            except Exception as e:
                st.error(f"Error fetching data for {sym}: {e}")

# ---------- CHART ----------
if results:
    ax.set_title("Actual vs Predicted Prices")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price ($)")
    ax.legend()
    st.pyplot(fig)

    # ---------- METRICS ----------
    st.subheader("📌 Key Metrics")
    metric_cols = st.columns(len(results))
    for col, sym in zip(metric_cols, results):
        r = results[sym]
        col.metric(f"{sym} Latest", f"${r['latest']:.2f}")
        col.metric(f"{sym} Predicted", f"${r['predicted_today']:.2f}")
        try:
            col.metric(f"🔜 {sym} Tomorrow", f"${float(r['tomorrow']):.2f}")
        except:
            col.metric(f"🔜 {sym} Tomorrow", "N/A")

    # ---------- DOWNLOAD SECTION ----------
    st.subheader("📥 Download Combined Predictions")
    combined_df = []
    for sym, r in results.items():
        df = r['df'].copy()
        df['Predicted'] = r['preds']
        df['Symbol'] = sym
        combined_df.append(df[['Symbol', 'Close', 'Predicted']])
    final_df = pd.concat(combined_df).reset_index().rename(columns={"index": "Date"})

    csv = final_df.to_csv(index=False).encode()
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="asset_predictions.csv",
        mime='text/csv'
    )

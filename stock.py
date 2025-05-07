import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# Function to fetch stock data using yfinance
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")  # Fetch last 1 year of data
    return hist

# Function to preprocess stock data
def preprocess_stock_data(data):
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = pd.to_datetime(df.index)
    return df

# Function to build and train the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the ARIMA model
def train_arima_model(train_data):
    model = ARIMA(train_data, order=(5, 1, 0))  # Adjust the ARIMA order as needed
    model_fit = model.fit()
    return model_fit

# Streamlit App
st.set_page_config(page_title="Indian Stock Price Prediction", layout="wide")
st.title("Indian Stock Price Prediction")

# Sidebar for Indian Stock Symbols
st.sidebar.title("Indian Stock Symbols")
indian_symbols = {
    'Reliance Industries': 'RELIANCE.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'Infosys': 'INFY.NS',
    'Bharti Airtel': 'AIRTEL.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Larsen & Toubro': 'LT.NS',
    'State Bank of India': 'SBIN.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Wipro': 'WIPRO.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Dr. Reddy’s Laboratories': 'DRREDDY.NS',
    'Sun Pharmaceutical': 'SUNPHARMA.NS',
    'Titan Company': 'TITAN.NS',
    'Nestle India': 'NESTLEIND.NS',
}

# Display the stock symbols in the sidebar
for company, symbol in indian_symbols.items():
    st.sidebar.write(f"{company}: {symbol}")

# User input for stock symbol
symbol_input = st.text_input("Enter Stock Symbol (or select from the sidebar):", "")

# Button to show data
if st.button("Show Stock Data"):
    if symbol_input:
        # Fetch stock data for the input symbol
        data = fetch_stock_data(symbol_input)
        if not data.empty:
            df = preprocess_stock_data(data)

            # Display the last year of data
            st.subheader(f"Stock Data for {symbol_input}")
            st.write(df.head(10))

            # Display the current closing price
            latest_price = df['Close'].iloc[-1]
            st.metric(label="Current Closing Price", value=f"₹{latest_price:.2f}")

            # Visualize the stock data (OHLC)
            st.subheader(f"Stock Price (OHLC) for {symbol_input}")
            fig, ax = plt.subplots(figsize=(12, 6))
            df[['Open', 'High', 'Low', 'Close']].plot(ax=ax)
            ax.set_title(f"{symbol_input} - OHLC Price")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (INR)')
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(fig)

            # Extract and display data for each month in the last year
            df['Month'] = df.index.month
            df['Year'] = df.index.year

            # Create columns for each month in the last year
            current_year = datetime.now().year
            for month in range(1, 13):  # Loop through all months from Jan to Dec
                month_data = df[(df['Month'] == month) & (df['Year'] == current_year)]
                if not month_data.empty:
                    month_name = datetime(2000, month, 1).strftime('%B')

                    # Display month data and chart side-by-side
                    col1, col2 = st.columns([1, 2])  # Define two columns: one for data, one for visualization

                    # Month data in the left column
                    with col1:
                        st.subheader(f"{month_name} {current_year} Stock Prices")
                        st.write(month_data[['Open', 'High', 'Low', 'Close']])

                    # Month visualization in the right column
                    with col2:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        month_data[['Open', 'High', 'Low', 'Close']].plot(ax=ax)
                        ax.set_title(f"{symbol_input} - {month_name} Price Action")
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Price (INR)')
                        plt.xticks(rotation=45)
                        plt.grid(True)
                        st.pyplot(fig)

            # Prepare data for training LSTM and ARIMA
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Close']].values)

            X_train = []
            y_train = []
            for i in range(60, len(scaled_data)):
                X_train.append(scaled_data[i - 60:i, 0])
                y_train.append(scaled_data[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # Train LSTM Model
            lstm_model = build_lstm_model(X_train.shape)
            lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

            # Train ARIMA Model
            arima_model = train_arima_model(df['Close'])

            # Prepare data for prediction
            inputs = scaled_data[-60:]
            inputs = np.reshape(inputs, (1, inputs.shape[0], 1))

            # LSTM Prediction
            predicted_price_lstm = lstm_model.predict(inputs)
            predicted_price_lstm = scaler.inverse_transform(predicted_price_lstm)[0][0]

            # ARIMA Prediction
            arima_forecast = arima_model.forecast(steps=1)
            arima_predicted_price = arima_forecast.iloc[0]

            # Ensemble Prediction (Weighted Average of LSTM and ARIMA)
            weight_lstm = 0.6  # You can adjust weights based on model performance
            weight_arima = 0.4
            ensemble_prediction = (weight_lstm * predicted_price_lstm) + (weight_arima * arima_predicted_price)

            # Display predictions
            st.subheader(f"Predicted Stock Price for {symbol_input}")
            st.write(f"LSTM Predicted Price: ₹{predicted_price_lstm:.2f}")
            st.write(f"ARIMA Predicted Price: ₹{arima_predicted_price:.2f}")
            st.write(f"Ensemble Predicted Price: ₹{ensemble_prediction:.2f}")

        else:
            st.error("Error fetching data. Please check the stock symbol.")
    else:
        st.warning("Please enter a stock symbol.")

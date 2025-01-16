
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Streamlit page configuration (must be at the top)
st.set_page_config(layout="wide", page_title="Stock Trend Prediction")

# Function to add a background image
def add_dynamic_bg(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;

        }}
         
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background image (Stock Market-related)
stock_market_image_url = "https://c0.wallpaperflare.com/preview/347/508/926/analysis-analytics-analyzing-business.jpg"
add_dynamic_bg(stock_market_image_url)

# Streamlit App Title
st.title('📈 Stock Trend Prediction App')

# App description
st.write("""
This app predicts stock trends using a pre-trained deep learning model. 
Explore historical data, visualize trends, and view predicted vs original prices.
""")

# Sidebar for user input
st.sidebar.header('User Input')
user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
date_range = st.sidebar.date_input("Select Date Range", [pd.to_datetime("2010-01-01"), pd.to_datetime("2023-12-31")])

# Fetch stock data
try:
    start, end = date_range
    st.sidebar.write(f"Fetching data from {start} to {end}")
    df = yf.download(user_input, start=start, end=end)

    # Data Description
    st.subheader(f"Data for {user_input} from {start} to {end}")
    if st.checkbox("Show Raw Data"):
        st.write(df)

    # Visualization: Closing Price vs Time
    st.subheader(f"Closing Price vs Time for {user_input}")
    fig = plt.figure(figsize=(8,4))
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.title(f'Closing Price vs Time for {user_input}')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.grid()
    plt.legend()
    st.pyplot(fig)

    # Visualization: Closing Price with 100MA
    ma100 = df['Close'].rolling(100).mean()
    st.subheader('Closing Price with 100-Day Moving Average')
    fig = plt.figure(figsize=(8,4))
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.plot(ma100, label='100-Day MA', color='orange')
    plt.title(f'Closing Price and 100-Day MA for {user_input}')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.grid()
    plt.legend()
    st.pyplot(fig)

    # Visualization: Closing Price with 100MA & 200MA
    ma200 = df['Close'].rolling(200).mean()
    st.subheader('Closing Price with 100-Day & 200-Day Moving Averages')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.plot(ma100, label='100-Day MA', color='orange')
    plt.plot(ma200, label='200-Day MA', color='green')
    plt.title(f'Closing Price with 100MA & 200MA for {user_input}')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.grid()
    plt.legend()
    st.pyplot(fig)

    # Splitting Data into Training and Testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load pre-trained model
    model = load_model('keras_model.h5')

    # Prepare testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(np.array(final_df).reshape(-1, 1))

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Predictions
    y_predicted = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Visualization: Predictions vs Original
    st.subheader('Predicted vs Original Prices')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test, label='Original Price', color='green')
    plt.plot(y_predicted, label='Predicted Price', color='red')
    plt.title(f'Predicted vs Original Prices for {user_input}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid()
    plt.legend()
    st.pyplot(fig2)

except Exception as e:
    st.error(f"An error occurred: {e}")





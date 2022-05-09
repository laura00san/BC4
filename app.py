

import streamlit as st
from datetime import date

import yfinance as yf
from plotly import graph_objs as go

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.neural_network import MLPRegressor


START = "2019-05-05"
#current date
TODAY= date.today().strftime("%Y-%m-%d")  

#create APP
#APP title
st.title("Stock Prediction App")
#Different choices of used crypto currencies
stocks = ("BTC-USD", "ADA-USD", "AVAX-USD", "AXS-USD", "ETH-USD", 
"ATOM-USD ", "LINK-USD", "LUNA1-USD", "MATIC-USD", "SOL-USD")


#create select box 
selected_stock = st.selectbox("Select dataset for predicition", stocks)

#create slider to select nº of years fro prediction
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

#cache the data of the stock we choose and doesn't have to run the code below again
@st.cache 
def load_data(ticker):
    data= yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

st.subheader('Raw data')
st.write(data.tail())

@st.cache 
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name= 'stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name= 'stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()



scaler = MinMaxScaler()
@st.cache 
def predict_data(ticker):
    data_1 = data.set_index('Date')
    y_data = data_1["Close"]
    X_data = data_1.loc[:, data_1.columns != 'Close']   
    X_data_scaled = pd.DataFrame(scaler.fit_transform(X_data), index= X_data.index, columns = X_data.columns)


    X_data_train = X_data_scaled[:int(X_data_scaled.shape[0]*0.8)]
    X_data_test = X_data_scaled[int(X_data_scaled.shape[0]*0.8):]
    y_data_train = y_data[:int(X_data_scaled.shape[0]*0.8)]
    y_data_test = y_data[int(X_data_scaled.shape[0]*0.8):]

    model = MLPRegressor(hidden_layer_sizes=100, max_iter=100,activation = 'relu',solver='lbfgs',random_state=1).fit(X_data_train, y_data_train)
    y_pred = model.predict(X_data_test)
    data_pred = y_pred
    tabela_previsões = pd.DataFrame(data_pred)
    tabela_previsões['Date'] = X_data_test.index
    tabela_previsões = tabela_previsões.set_index('Date')
    tabela_previsões.rename(columns= {0:'Close Price Prediction'}, inplace=True) 
    return tabela_previsões

data_pred = predict_data(selected_stock)

data_load_state.text("Loading forecasting data... done!")
st.subheader('Forecast data')
st.write(data_pred.tail())


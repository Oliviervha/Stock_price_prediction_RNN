# Stock price prediction using RNN
A Recurrent Neural Network using time series data for stock price prediction.

## Data
To train the stock price prediction model the stock closing prices for the S&P500 were considered. 

## Model
A Recurrent Neural Network (RNN) is a type of neural network well-suited to time series data. To develop and train the model Tensorflow have been used in Python.  

## Evaluation
A test set has been used for evaluation. Here, each three days, the stock closing price for the next three days was predicted by the model. 
* MAE: 55.79
* RMSE: 67.67

![alt text](https://github.com/Oliviervha/Stock_price_prediction_RNN/blob/main/Predictions.png?raw=true)

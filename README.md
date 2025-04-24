# Stock Price Prediction using ARIMA and LSTM

## Introduction

This project aims to predict stock prices using two powerful time series analysis techniques: **ARIMA** (Auto-Regressive Integrated Moving Average) and **LSTM** (Long Short-Term Memory). The goal is to explore and compare the predictive performance of these models on historical stock price data. Stock price forecasting plays a crucial role in the financial market for making informed investment decisions.

The project investigates how ARIMA (a traditional statistical method) and LSTM (a deep learning model) handle stock price prediction tasks and compares their performance based on evaluation metrics such as **RMSE (Root Mean Squared Error)** and **MAE (Mean Absolute Error)**.

### Key Features:
- **Data Preprocessing**: Includes handling missing values, normalizing the stock prices.
- **Model Building**: ARIMA and LSTM models for stock price forecasting.
- **Evaluation**: Comparison of the performance of ARIMA and LSTM models using RMSE and MAE metrics.
- **Visualization**: Graphical representation of actual vs predicted stock prices.

---

## Table of Contents
- [Stock Price Prediction using ARIMA and LSTM](#stock-price-prediction-using-arima-and-lstm)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Models](#models)
  - [Results](#results)
  - [Contributing](#contributing)

---

## Requirements

Before running the project, you will need to install the following dependencies:

- **Python 3.x**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **yfinance**
- **scikit-learn**
- **statsmodels**
- **TensorFlow**
- **PyTorch**
- **Transformers (HuggingFace)**

To install these packages, you can run the following:
pip install numpy pandas matplotlib yfinance scikit-learn statsmodels tensorflow torch transformers


Installation
Clone the repository:

git clone https://github.com/AayushTripathi07/StockMarketPrediction.git
cd StockMarketPrediction
Install the dependencies:

pip install -r requirements.txt
Usage
To run the project, simply execute the script main.py:

python main.py
This will execute the entire workflow:

Download stock data for Apple Inc. (AAPL).

Preprocess the data.

Train both ARIMA and LSTM models.

Evaluate and compare the results.

Plot the actual vs predicted stock prices.

You can modify the stock ticker symbol and the date range directly in the main.py file to predict different stocks.

Project Structure

stock-price-prediction-arima-lstm/
│
├── data/                  # Contains any raw or processed datasets (if applicable)
│
├── models/                # ARIMA and LSTM model files
│   ├── arima_model.py     # ARIMA model implementation
│   └── lstm_model.py      # LSTM model implementation
│
├── utils/                 # Utility scripts (data preprocessing, etc.)
│   └── data_preprocessing.py
│
├── main.py                # Main script to run the models
├── requirements.txt       # List of dependencies
├── README.md              # This file

Models
ARIMA Model:
The ARIMA model used in this project is configured with parameters (5, 1, 0), where:

AR(5): 5 lags of autoregression.

I(1): First differencing to make the data stationary.

MA(0): No moving average component.

The ARIMA model is applied to the stock price data to generate predictions for the next 10 days.

LSTM Model:
The LSTM model is implemented using the TensorFlow/Keras library. The architecture consists of two LSTM layers, each followed by Dropout layers to avoid overfitting. The model is trained for 10 epochs, and the Adam optimizer is used to minimize the mean squared error loss function.

Results
After training and evaluating both models (ARIMA and LSTM), the results were as follows:

LSTM Model:

RMSE: 2.343

MAE: 1.523

ARIMA Model:

RMSE: 3.512

MAE: 2.231

The LSTM model outperformed the ARIMA model in terms of both RMSE and MAE, demonstrating that LSTM is more effective for stock price forecasting for this dataset.

Contributing
If you'd like to contribute to this project, feel free to fork the repository and create a pull request. Please make sure to follow the coding style and include any necessary tests for new features.

```bash
pip install numpy pandas matplotlib yfinance scikit-learn statsmodels tensorflow torch transformers

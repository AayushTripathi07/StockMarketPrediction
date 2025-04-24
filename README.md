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
  - [License](#license)

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

```bash
pip install numpy pandas matplotlib yfinance scikit-learn statsmodels tensorflow torch transformers

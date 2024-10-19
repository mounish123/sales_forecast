import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
data = pd.read_csv('C:\\Users\\mouni\\downloads\\sales_dataset\\Amazon_Sale_Report.csv', low_memory=False)

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  

if data['Date'].isnull().any():
    print("There are NaT values in the Date column.")
    data = data[data['Date'].notna()]  

data.set_index('Date', inplace=True)
daily_sales = data.groupby(data.index).sum()  # Adjust if you need specific columns

# Fit the ARIMA model
model = ARIMA(daily_sales['Amount'], order=(2, 1, 2))  
model_fit = model.fit()

# Forecasting
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(daily_sales['Amount'], label='Historical Sales', color='blue') 
plt.plot(forecast_series, label='Forecast', color='red', marker='o')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

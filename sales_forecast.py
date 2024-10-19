# Install required libraries if you haven't already
# !pip install pandas numpy matplotlib statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Create synthetic sales data
np.random.seed(42)  # For reproducibility
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
sales = np.random.poisson(lam=200, size=len(dates)) + np.linspace(0, 50, len(dates))

# Create a DataFrame
data = pd.DataFrame({'date': dates, 'sales': sales})
data.set_index('date', inplace=True)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(data['sales'], label='Sales Over Time')
plt.title('Synthetic Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Fit the ARIMA model (you can adjust the (p, d, q) parameters)
model = ARIMA(data['sales'], order=(2, 1, 2))  # (p, d, q)
model_fit = model.fit()

# Print the summary of the model
print(model_fit.summary())

# Forecasting
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(data['sales'], label='Historical Sales', color='blue')
plt.plot(forecast_series, label='Forecast', color='red', marker='o')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# If you had actual sales data for comparison, you could calculate MSE:
# actual_values = [your_actual_values]  # Replace with actual data
# mse = mean_squared_error(actual_values, forecast)
# print(f'Mean Squared Error: {mse}')

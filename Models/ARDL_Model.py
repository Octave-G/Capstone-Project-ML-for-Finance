import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.ardl import ARDL
import matplotlib.pyplot as plt
import pandas as pd

#                           Data
data = pd.read_csv('data_ardl.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date', ascending=True)
data = data.set_index('Date')
data.dropna(inplace=True)

exog = data.drop('US T-Note 10Y Yield', axis=1) # Exogenous data
endog = data['US T-Note 10Y Yield'] # Endogenous data is 10Y U.S. T-Note

print('\t Data Head :', '\n', data.head(), '\n')
print('Data Shape :', data.shape)


# AutoCorrelation, Partial Autocorrelation of Target
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(data['US T-Note 10Y Yield'], ax=axes[0], lags=20)
plot_pacf(data['US T-Note 10Y Yield'], ax=axes[1], lags=20, method='ywm') # 'ywm' or 'ols'
plt.show()


# Augmented Dickey-Fuller Test
print('\t Augmented Dickey-Fuller Test :', '\n',)
for feature in data.columns :
    result = adfuller(data[feature].dropna(axis=0))
    print(f'- {feature} :', "Stationary" if result[1] <= 0.05 else "Non-Stationary") # Reject or not null hypothesis H0
    print(f'ADF Statistic: {round(result[0], 3)}, p-value: {round(result[1], 3)} \n')


# Train, Test Split
split_point = int(len(endog) * 0.75) # 75% Train, 25% Test
exog_train, exog_test = exog[:split_point], exog[split_point:]
endog_train, endog_test = endog[:split_point], endog[split_point:]

n, p = exog_test.shape[0], exog_test.shape[1] # NB of samples and features in test set (Useful for R2 Adjusted)


#                           ARDL
# Model
ARDL_model = ARDL(endog=endog_train, lags=2, exog=exog_train, order=1, causal=True, trend='c')
results = ARDL_model.fit()
forecast = results.predict(start=endog_test.index[0], end=endog_test.index[-1], exog_oos=exog_test)

# Metrics
ARDL_MSE = mean_squared_error(endog_test, forecast)
ARDL_R2 = r2_score(endog_test, forecast)
ARDL_R2_ADJ = 1 - (((1 - ARDL_R2) * (n - 1)) / (n - p - 1))

print('\tARDL Forecasting Metrics :', '\n', f'\nMSE = {round(ARDL_MSE,3)} \nR2 = {round(ARDL_R2,3)} \nAdjusted R2 = {round(ARDL_R2_ADJ,3)}\n')

print(results.summary())

# Plot alongside historical data
plt.figure(figsize=(12,6))
plt.plot(data.index, data['US T-Note 10Y Yield'], label='Historical Data', color='blue', alpha=0.4)
plt.plot(forecast.index, forecast, label='Prediction', color='red', alpha=1)
plt.title('ARDL Prediction on 10Y T-Bills'); plt.xlabel('Date'); plt.ylabel('Log Differenciation of US 10Y Yield'); plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# Plot with 95% confidence intervals
results.plot_predict(start=forecast.index[0], end=forecast.index[-1], dynamic=False, exog_oos=exog_test,)
plt.title('ARDL Prediction on 10Y T-Bills with Confidence Interval'); plt.xlabel('Date'); plt.ylabel('Log Differenciation of US 10Y Yield'); plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# Diagnostic Check
print(results.diagnostic_summary())
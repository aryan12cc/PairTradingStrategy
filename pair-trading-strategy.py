'''
Steps to initiate the pair trading strategy:
1. Check if both the stocks are correlated and cointegrated
    a. The correlation should be above a certain threshold (say 0.95)
    b. The p-value of ADF (Augmented Dickey-Fuller) test should be less than 0.05
2. Calculate the spread between the two stocks
3. Calculate the z-score of the spread (standard score)
    a. The z-score has to be mean reverting since the stocks are cointegrated
4.  a. If the z-score is > +2, then short the dependent stock and long the independent stock
    b. If the z-score is < -2, then long the dependent stock and short the independent stock
5. Apply hedging to reduce the risk
6. Also apply some basic stock market signals
'''

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from scipy import stats

class PairTradingStrategy:
    def __init__(self):
        self.train_profits, self.test_profits = [], []
        self.train_dates, self.test_dates, self.dates = [], [], []

    def read_data(self):
        # reading mastercard and visa data
        mastercard = pd.read_csv('data/mastercard.csv')    
        visa = pd.read_csv('data/visa.csv')
        # resetting the index
        mastercard.reset_index(drop = True, inplace = True)
        visa.reset_index(drop = True, inplace = True)
        # checking only the dates that are common in both the dataframes
        mastercard = mastercard[mastercard['Date'].isin(visa['Date'])]
        visa = visa[visa['Date'].isin(mastercard['Date'])]
        print(mastercard.describe())
        print(visa.describe())
        # reading adjusted closing values for mastercard and visa
        mastercard_closing_values = mastercard['Adj Close']
        visa_closing_values = visa['Adj Close']
        # storing dates for pyplot
        self.dates = mastercard['Date']
        return mastercard_closing_values, visa_closing_values
    
    def split_data(self, mastercard_closing_values, visa_closing_values):
        # splitting the data into training and testing data (65:35 ratio)
        train_test_split_ratio = 0.65
        train_size = int(train_test_split_ratio * len(mastercard_closing_values))
        mastercard_train = mastercard_closing_values[:train_size]
        visa_train = visa_closing_values[:train_size]
        mastercard_test = mastercard_closing_values[train_size:]
        visa_test = visa_closing_values[train_size:]
        # storing dates for pyplot
        self.train_dates = pd.to_datetime(self.dates[:train_size])
        self.test_dates = pd.to_datetime(self.dates[train_size:])
        return mastercard_train, visa_train, mastercard_test, visa_test
    
    def find_correlation(self, mastercard_data, visa_data):
        # finding the correlation between the stock prices of mastercard and visa data
        correlation = np.corrcoef(mastercard_data, visa_data)[0][1]
        print(f'correlation = {correlation}')
        # checking if the correlation is high enough for pair trading
        # if correlation < 0.95:
        #     print(f'The correlation is {correlation}. The pair of stocks is not suitable for pair trading')
        #     sys.exit()
        return correlation
    
    def find_p_value(self, mastercard_data, visa_data):
        # finding the p-value of the ADF test for co-integration
        result = coint(mastercard_data, visa_data)
        p_value = result[1]
        # checking if the p-value is low enough for pair trading
        if p_value > 0.1:
            print(f'The p-value is {p_value}. The pair of stocks is not suitable for pair trading')
            sys.exit()
        return p_value
    
    def regression(self, mastercard_data, visa_data):
        # drawing a linear regression model between the two stocks
        model = sm.OLS(list(mastercard_data), list(visa_data)).fit()
        return model
    
    def generate_residuals(self, mastercard_data, visa_data, hedge_ratio):
        # generating residuals
        residuals = mastercard_data - hedge_ratio * visa_data
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        return residuals, residual_mean, residual_std
    
    def train(self, mastercard_data, visa_data):
        correlation = self.find_correlation(mastercard_data, visa_data)
        coint_p_value = self.find_p_value(mastercard_data, visa_data)
        print(f'correlation = {correlation}')
        print(f'cointegration p-value = {coint_p_value}')
        model = self.regression(mastercard_data, visa_data)
        print(model.summary())
        # getting the hedge ratio from the model. 
        # this tells us how many stocks of visa we should buy for each stock of mastercard
        hedge_ratio = model.params[0]
        print(f'hedge ratio: {hedge_ratio}')
        return hedge_ratio
    
    def generate_z_score(self, mastercard_data, visa_data, hedge_ratio, dates_used):
        residuals, residual_mean, residual_std = self.generate_residuals(mastercard_data, visa_data, hedge_ratio)
        z_score = (residuals - residual_mean) / residual_std

        plt.figure(figsize = (10, 5))
        plt.plot(dates_used, z_score)
        if list(dates_used) == list(self.train_dates):
            plt.title('Z-score of training data')
        else:
            plt.title('Z-score of testing data')
        plt.xlabel('Dates')
        plt.xticks(rotation = 45, ha = 'right')
        plt.ylabel('Z-score')
        plt.axhline(1, color = 'red', linestyle = '--')
        plt.axhline(0, color = 'black', linestyle = '-')
        plt.axhline(-1, color = 'red', linestyle = '--')
        plt.show()

        

def main():
    obj = PairTradingStrategy()
    mastercard_closing_values, visa_closing_values = obj.read_data()
    mastercard_train, visa_train, mastercard_test, visa_test = obj.split_data(mastercard_closing_values, visa_closing_values)

    hedge_ratio = obj.train(mastercard_train, visa_train)
    obj.generate_z_score(mastercard_train, visa_train, hedge_ratio, obj.train_dates)

    obj.generate_z_score(mastercard_test, visa_test, hedge_ratio, obj.test_dates)
    

if __name__ == "__main__":
    main()
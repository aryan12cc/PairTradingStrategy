import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from scipy import stats

def plot_data(x, y, title, xtitle, ytitle, horizontal_lines):
    plt.figure(figsize = (10, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval = 6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    if horizontal_lines == True:
        plt.axhline(0.75, color = 'red', linestyle = '--')
        plt.axhline(0, color = 'black', linestyle = '-')
        plt.axhline(-0.75, color = 'red', linestyle = '--')
    plt.show()

class PairTradingStrategy:
    def __init__(self):
        self.mastercard_train, self.visa_train = [], []
        self.mastercard_test, self.visa_test = [], []
        self.mastercard_db = pd.DataFrame()
        self.visa_db = pd.DataFrame()
        self.profits = []
        self.dates_train, self.dates_test = [], []
        self.z_score = []
        self.hedge_ratio = 0

    def read_data(self):
        # reading mastercard and visa data
        self.mastercard_db = pd.read_csv('data/mastercard.csv')
        self.visa_db = pd.read_csv('data/visa.csv') 
        # checking only the dates that are common in both the dataframes
        self.mastercard_db = self.mastercard_db[self.mastercard_db['Date'].isin(self.visa_db['Date'])]
        self.visa_db = self.visa_db[self.visa_db['Date'].isin(self.mastercard_db['Date'])]
        # reading date and adjusted closing values for mastercard and visa
        self.mastercard_db = self.mastercard_db[['Date', 'Adj Close']]
        self.visa_db = self.visa_db[['Date', 'Adj Close']]
    
    def split_data(self):
        # splitting the data into training and testing data (65:35 ratio)
        train_test_split_ratio = 0.65
        train_size = int(train_test_split_ratio * len(self.mastercard_db['Adj Close']))
        self.mastercard_train = self.mastercard_db['Adj Close'][:train_size]
        self.visa_train = self.visa_db['Adj Close'][:train_size]
        self.mastercard_test = self.mastercard_db['Adj Close'][train_size:]
        self.visa_test = self.visa_db['Adj Close'][train_size:]
        # resetting the index for the testing data
        self.mastercard_test.reset_index(drop = True, inplace = True)
        self.visa_test.reset_index(drop = True, inplace = True)
        # storing dates for pyplot
        self.dates_train = self.mastercard_db['Date'][:train_size]
        self.dates_test = self.mastercard_db['Date'][train_size:]
        # parsing dates
        self.dates_train = pd.to_datetime(self.dates_train)
        self.dates_test = pd.to_datetime(self.dates_test)
    
    def find_correlation(self, mastercard_data, visa_data):
        # finding the correlation between the stock prices of mastercard and visa data
        correlation = np.corrcoef(mastercard_data, visa_data)[0][1]
        print(f'correlation: {correlation}')
        # checking if the correlation is high enough for pair trading
        if correlation < 0.9:
            print(f'The correlation is {correlation}. The pair of stocks is not suitable for pair trading')
            sys.exit()
        return correlation
    
    def find_p_value(self, mastercard_data, visa_data):
        # finding the p-value of the ADF test for co-integration
        result = coint(mastercard_data, visa_data)
        p_value = result[1]
        print(f'p-value: {p_value}')
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
    
    def train(self):
        correlation = self.find_correlation(self.mastercard_train, self.visa_train)
        coint_p_value = self.find_p_value(self.mastercard_train, self.visa_train)
        # getting the hedge ratio from the model
        # this tells us how many stocks of visa we should buy for each stock of mastercard
        model = self.regression(self.mastercard_train, self.visa_train)
        self.hedge_ratio = model.params[0]
        print(f'hedge ratio: {self.hedge_ratio}')
    
    def generate_z_score(self, mastercard_data, visa_data, dates):
        residuals, residual_mean, residual_std = self.generate_residuals(mastercard_data, visa_data, self.hedge_ratio)
        self.z_score = (residuals - residual_mean) / residual_std
    
    def get_profit(self, current_balance, mastercard_stocks, visa_stocks, mastercard_price, visa_price):
        return current_balance + (mastercard_stocks * mastercard_price) + (visa_stocks * visa_price)

    def check_trade(self, mastercard_stocks, visa_stocks, mastercard_data, visa_data, current_iter, buy_stock_name):
        # checking if trade is possible without exceeding the limit of max stocks
        if buy_stock_name == 'Visa':
            if abs(mastercard_stocks - 1) > 100 or abs(visa_stocks + mastercard_data[current_iter] / visa_data[current_iter]) > 100:
                return False
            return True
        else:
            if abs(mastercard_stocks + 1) > 100 or abs(visa_stocks - mastercard_data[current_iter] / visa_data[current_iter]) > 100:
                return False
            return True

    def simulate_trading(self, mastercard_data, visa_data):
        mastercard_stocks = 0
        visa_stocks = 0
        current_balance = 0
        # current_capital_invested keeps the total money required to start the trade
        # since the trade is a hedged trade, current_capital_invested is the money required to go long
        # we also add a margin of 20%
        current_capital_invested = 0
        max_capital_invested = 0
        z_score_list = list(self.z_score)
        self.profits.clear()
        for i in range(len(mastercard_data)):
            # if the z-score is > 0.75, then short mastercard, long visa
            if z_score_list[i] > 0.75:
                if self.check_trade(mastercard_stocks, visa_stocks, mastercard_data, visa_data, i, 'Mastercard') == True:
                    mastercard_stocks -= 1
                    visa_stocks += self.hedge_ratio
                    current_balance += (mastercard_data[i]) - (visa_data[i] * self.hedge_ratio)
                    current_capital_invested -= 1.2 * mastercard_data[i]
            # if the z-score is < -0.75, then long mastercard, short visa
            elif z_score_list[i] < -0.75:
                if self.check_trade(mastercard_stocks, visa_stocks, mastercard_data, visa_data, i, 'Visa') == True:
                    mastercard_stocks += 1
                    visa_stocks -= self.hedge_ratio
                    current_balance += (visa_data[i] * self.hedge_ratio) - (mastercard_data[i])
                    current_capital_invested += 1.2 * mastercard_data[i]
            # if the z-score is between -0.25 and 0.25, we close the positions
            # if the z-score is beyond -3 and 3, we trigger stop loss
            # if it is the last day of the market, we even our position
            if (-0.25 < z_score_list[i] < 0.25) or (z_score_list[i] > 3 or z_score_list[i] < -3) or (i == len(mastercard_data) - 1):
                current_balance += mastercard_stocks * mastercard_data[i] + visa_stocks * visa_data[i]
                max_capital_invested = max(max_capital_invested, abs(current_capital_invested))
                current_capital_invested = 0
                mastercard_stocks = 0
                visa_stocks = 0
            # calculating profit
            current_profit = self.get_profit(current_balance, mastercard_stocks, visa_stocks, mastercard_data[i], visa_data[i])
            self.profits.append(current_profit)
        return self.profits, max_capital_invested

def main():
    obj = PairTradingStrategy()
    obj.read_data()
    obj.split_data()

    obj.train()
    obj.generate_z_score(obj.mastercard_train, obj.visa_train, obj.dates_train)
    plot_data(obj.dates_train, obj.z_score, 'Training Data Z-Score', 'Date', 'Z-Score', True)

    train_profits, train_max_capital_invested = obj.simulate_trading(obj.mastercard_train, obj.visa_train)
    plot_data(obj.dates_train, train_profits, 'Training Data Value', 'Date', 'Total Value in USD', False)
    print(f'Training data profits: ${train_profits[-1]}')
    print(f'Training data return value: {train_profits[-1] / train_max_capital_invested * 100}%')

    obj.generate_z_score(obj.mastercard_test, obj.visa_test, obj.dates_test)
    plot_data(obj.dates_test, obj.z_score, 'Testing Data Z-Score', 'Date', 'Z-Score', True)

    test_profits, test_max_capital_invested = obj.simulate_trading(obj.mastercard_test, obj.visa_test)
    plot_data(obj.dates_test, test_profits, 'Testing Data Value', 'Date', 'Total Value in USD', False)
    print(f'Testing data profits: ${test_profits[-1]}')
    print(f'Testing data return value: {test_profits[-1] / test_max_capital_invested * 100}%')
    

if __name__ == "__main__":
    main()
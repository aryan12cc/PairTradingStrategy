import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from scipy import stats

def plot_data(x, y, title, xtitle, ytitle, horizontal_lines, mastercard_short_sd = 0, mastercard_long_sd = 0, closing_position_sd = 0, stop_loss_sd = 0):
    plt.figure(figsize = (10, 5))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval = 6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    if horizontal_lines == True:
        plt.axhline(mastercard_short_sd, color = 'red', linestyle = '--')
        plt.axhline(closing_position_sd, color = 'green', linestyle = '--')
        plt.axhline(stop_loss_sd, color = 'orange', linestyle = '--')
        plt.axhline(0, color = 'black', linestyle = '-')
        plt.axhline(mastercard_long_sd, color = 'red', linestyle = '--')
        plt.axhline(-closing_position_sd, color = 'green', linestyle = '--')
        plt.axhline(-stop_loss_sd, color = 'orange', linestyle = '--')
        plt.legend(['Z-Score', 'Short / Long Market Signal', 'Exit Position', 'Stop Loss'])
    plt.show()

class PairTradingStrategy:
    # mastercard_short_sd refers to the standard deviation at which we will short mastercard
    # mastercard_long_sd refers to the standard deviation at which we will long mastercard
    # closing_position_sd refers to the lower bound of the z-score at which we will close the position
    # stop_loss_sd refers to the upper bound of the z-score at which we will trigger stop loss
    def __init__(self, mastercard_short_sd, mastercard_long_sd, closing_position_sd, stop_loss_sd):
        self.mastercard_train, self.visa_train = [], []
        self.mastercard_test, self.visa_test = [], []
        self.mastercard_db = pd.DataFrame()
        self.visa_db = pd.DataFrame()
        self.profits = []
        self.max_capital_invested = []
        self.dates_train, self.dates_test = [], []
        self.z_score = []
        self.hedge_ratio = 0
        self.mastercard_short_sd = mastercard_short_sd
        self.mastercard_long_sd = mastercard_long_sd
        self.closing_position_sd = closing_position_sd
        self.stop_loss_sd = stop_loss_sd
        self.margin_value = 1.2

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
        # storing dates for pyplot
        self.dates_train = self.mastercard_db['Date'][:train_size]
        self.dates_test = self.mastercard_db['Date'][train_size:]
        # parsing dates
        self.dates_train = pd.to_datetime(self.dates_train)
        self.dates_test = pd.to_datetime(self.dates_test)
        # resetting the index for the testing data
        self.mastercard_test.reset_index(drop = True, inplace = True)
        self.visa_test.reset_index(drop = True, inplace = True)
        self.dates_test.reset_index(drop = True, inplace = True)
    
    def find_correlation(self, mastercard_data, visa_data):
        # finding the correlation between the stock prices of mastercard and visa data
        correlation = np.corrcoef(mastercard_data, visa_data)[0][1]
        print(f'  correlation: {correlation}')
        # checking if the correlation is high enough for pair trading
        if correlation < 0.9:
            print(f'  The correlation is {correlation}. The pair of stocks is not suitable for pair trading')
            sys.exit()
        return correlation
    
    def find_p_value(self, mastercard_data, visa_data):
        # finding the p-value of the ADF test for co-integration
        result = coint(mastercard_data, visa_data)
        p_value = result[1]
        print(f'  p-value: {p_value}')
        # checking if the p-value is low enough for pair trading
        if p_value > 0.1:
            print(f'  The p-value is {p_value}. The pair of stocks is not suitable for pair trading')
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
        print(f'  hedge ratio: {self.hedge_ratio}')
    
    def generate_z_score(self, mastercard_data, visa_data, dates):
        residuals, residual_mean, residual_std = self.generate_residuals(mastercard_data, visa_data, self.hedge_ratio)
        self.z_score = (residuals - residual_mean) / residual_std
    
    def get_profit(self, current_balance, mastercard_stocks, visa_stocks, mastercard_price, visa_price):
        return current_balance + (mastercard_stocks * mastercard_price) + (visa_stocks * visa_price)

    def check_trade(self, mastercard_stocks, visa_stocks, mastercard_data, visa_data, current_iter, buy_stock_name):
        # checking if trade is possible without exceeding the limit of max stocks
        if buy_stock_name == 'Mastercard':
            if abs(mastercard_stocks - 1) > 100 or abs(visa_stocks + mastercard_data[current_iter] / visa_data[current_iter]) > 100:
                return False
            return True
        else:
            if abs(mastercard_stocks + 1) > 100 or abs(visa_stocks - mastercard_data[current_iter] / visa_data[current_iter]) > 100:
                return False
            return True
    
    def check_year_gap(self, current_date, previous_date):
        # removing the time of the day
        current_date = str(current_date)[:10]
        previous_date = str(previous_date)[:10]
        
        # extracting the day, month and year
        current_year, current_month, current_day = int(current_date[:4]), int(current_date[5:7]), int(current_date[8:])
        previous_year, previous_month, previous_day = int(previous_date[:4]), int(previous_date[5:7]), int(previous_date[8:])

        # checking if the gap between the two dates is more than a year
        if current_year - previous_year > 1 or (current_year - previous_year == 1 and current_month > previous_month) or (current_year - previous_year == 1 and current_month == previous_month and current_day >= previous_day):
            return True
        return False

    def yearly_return_values(self, profits, max_capital_invested, dates):
        last_year_end_profit = 0
        last_year_start_index = 0
        for idx in range(len(self.profits)):
            if self.check_year_gap(dates[idx], dates[last_year_start_index]) == True:
                end_of_year_profit = profits[idx - 1]
                end_of_year_capital_invested = max_capital_invested[idx - 1]
                print(f'    Year {str(dates[idx])[:4]} return value: {round((end_of_year_profit - last_year_end_profit) / (end_of_year_capital_invested) * 100, 2)}%')
                last_year_end_profit = end_of_year_profit
                last_year_start_index = idx

    def simulate_trading(self, mastercard_data, visa_data):
        mastercard_stocks = 0
        visa_stocks = 0
        current_balance = 0
        # current_capital_invested keeps the total money required to start the trade
        # since the trade is a hedged trade, current_capital_invested is the money required to go long
        # we also add a margin of 20%
        current_capital_invested = 0
        z_score_list = list(self.z_score)
        self.profits.clear()
        self.max_capital_invested.clear()
        current_max_capital_invested = 0
        for i in range(len(mastercard_data)):
            # if the z-score is > mastercard_short standard deviation, then short mastercard, long visa
            if z_score_list[i] > self.mastercard_short_sd:
                if self.check_trade(mastercard_stocks, visa_stocks, mastercard_data, visa_data, i, 'Mastercard') == True:
                    mastercard_stocks -= 1
                    visa_stocks += mastercard_data[i] / visa_data[i]
                    current_capital_invested -= self.margin_value * mastercard_data[i]
                    current_max_capital_invested = max(current_max_capital_invested, abs(current_capital_invested))
            # if the z-score is < mastercard_long standard deviation, then long mastercard, short visa
            elif z_score_list[i] < self.mastercard_long_sd:
                if self.check_trade(mastercard_stocks, visa_stocks, mastercard_data, visa_data, i, 'Visa') == True:
                    mastercard_stocks += 1
                    visa_stocks -= mastercard_data[i] / visa_data[i]
                    current_capital_invested += self.margin_value * mastercard_data[i]
                    current_max_capital_invested = max(current_max_capital_invested, abs(current_capital_invested))
            # if the z-score is between closing position standard deviations, we close the positions
            # if the z-score is beyond stop loss standard deviations, we trigger stop loss
            # if it is the last day of the market, we even our position
            if (-self.closing_position_sd < z_score_list[i] < self.closing_position_sd) or (z_score_list[i] > self.stop_loss_sd or z_score_list[i] < -self.stop_loss_sd) or (i == len(mastercard_data) - 1):
                current_balance += mastercard_stocks * mastercard_data[i] + visa_stocks * visa_data[i]
                current_max_capital_invested = max(current_max_capital_invested, abs(current_capital_invested))
                current_capital_invested = 0
                mastercard_stocks = 0
                visa_stocks = 0
            # calculating profit
            current_profit = self.get_profit(current_balance, mastercard_stocks, visa_stocks, mastercard_data[i], visa_data[i])
            self.profits.append(current_profit)
            self.max_capital_invested.append(current_max_capital_invested)
        return self.profits, self.max_capital_invested

def main():
    # running multiple strategies with different market signal conditions
    pair_tradining_strategies = {
        'Strategy 1': PairTradingStrategy(0.75, -0.75, 0.25, 3),
        'Strategy 2': PairTradingStrategy(1, -1, 0.25, 3),
        'Strategy 3': PairTradingStrategy(0.75, -0.75, 0.25, 2.5),
        'Strategy 4': PairTradingStrategy(0.75, -0.75, 0.1, 3)
    }
    for strategy in pair_tradining_strategies.values():
        print('Current Strategy Values')
        print(f'Mastercard Short SD: {strategy.mastercard_short_sd}')
        print(f'Mastercard Long SD: {strategy.mastercard_long_sd}')
        print(f'Closing Position SD: {strategy.closing_position_sd}')
        print(f'Stop Loss SD: {strategy.stop_loss_sd}')
        strategy.read_data()
        strategy.split_data()

        strategy.train()
        strategy.generate_z_score(strategy.mastercard_train, strategy.visa_train, strategy.dates_train)
        plot_data(strategy.dates_train, strategy.z_score, 'Training Data Z-Score', 'Date', 'Z-Score', True, strategy.mastercard_short_sd, strategy.mastercard_long_sd, strategy.closing_position_sd, strategy.stop_loss_sd)

        train_profits, train_max_capital_invested = strategy.simulate_trading(strategy.mastercard_train, strategy.visa_train)
        plot_data(strategy.dates_train, train_profits, 'Training Data Value', 'Date', 'Total Value in USD', False)
        print(f'  Training data profits: ${round(train_profits[-1], 2)}')
        print(f'  Training data return values')
        strategy.yearly_return_values(train_profits, train_max_capital_invested, strategy.dates_train)
        print(f'    Training data return value: {round(train_profits[-1] / train_max_capital_invested[-1] * 100, 2)}%')

        strategy.generate_z_score(strategy.mastercard_test, strategy.visa_test, strategy.dates_test)
        plot_data(strategy.dates_test, strategy.z_score, 'Testing Data Z-Score', 'Date', 'Z-Score', True, strategy.mastercard_short_sd, strategy.mastercard_long_sd, strategy.closing_position_sd, strategy.stop_loss_sd)

        test_profits, test_max_capital_invested = strategy.simulate_trading(strategy.mastercard_test, strategy.visa_test)
        plot_data(strategy.dates_test, test_profits, 'Testing Data Value', 'Date', 'Total Value in USD', False)
        print(f'  Testing data profits: ${round(test_profits[-1], 2)}')
        print(f'  Testing data return values')
        strategy.yearly_return_values(test_profits, test_max_capital_invested, strategy.dates_test)
        print(f'    Testing data return value: {round(test_profits[-1] / test_max_capital_invested[-1] * 100, 2)}%')
    
if __name__ == "__main__":
    main()
from ZWG_pm import strat_function
from backtest import backtest

# Data that is passed in must fit the backtesting format (dataframe of prices with datetime as index, each column name is a stock ticker)
backtest(strat_function, 10000, 'portfolio_allocation/backtest_v2/price_data_cleaned.csv',
         'portfolio_allocation/backtest_v2/price_data_cleaned.csv', True, "portfolio_allocation/backtest_v2/ZWG_pm_results.csv")

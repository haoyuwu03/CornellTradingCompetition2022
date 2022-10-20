from backtest import backtest
import numpy as np
import pandas as pd


class Stocks():
    '''
    __init__(self, path)- Initializes stock information from historical data
    :argument: path - path to historical data of stocks 
    '''
    def __init__(self, path) -> None:
        # (Dataframe) Historical stock prices with datetime object as indices and stock ticker as each column's name 
        self.price_data = pd.read_csv(path)

        # (Dataframe) Historical stock returns with datetime object as indices and stock ticker as each column's name 
        self.returns_data = None

        # (Boolean) Whether or not to calculate historical statistics (updated after first day to False)
        self.first_run = True

        # (Series) Annual returns of each stock with stock ticker labels (EMA updated daily)
        self.annual_returns = None

        # (Series) Variance of each stock with stock ticker labels (EMA updated daily)
        self.vars = None

        # (Float) Average variance across all the stocks during historical
        self.avg_var = None

        # (Series) Mean price of each stock with stock ticker labels to calculate variance (EMA updated daily)
        self.mean_prices = None

        # (Series) Last price of each stock to calculate new return (updated daily)
        self.last_prices = None

        # (Dataframe) Covariance matrix between stocks (diagonal is updated with new variances)
        self.cov = None

        # (List) Tickers to label stock information Series's
        self.tickers = ['YPA', 'AOX', 'KWE', 'VSR', 'VWR', 'ZNX', 'MWN', 'MZJ', 'VEO', 'GJA',
                        'TUW', 'HBM', 'WNF', 'ABM', 'RLR', 'GPE', 'BUL', 'PKU', 'VLK', 'VYQ',
                        'EVW', 'NQW', 'PHV', 'DAI', 'WGQ', 'JFL', 'RLF', 'NFR', 'ZKY', 'STK',
                        'OWG', 'YLF', 'CPM', 'TGI', 'MGW', 'NTR', 'KKD', 'JPN', 'CGS', 'VOX']
        # (List) Labels for weights returned from strategy function to fit backtester's expected format
        self.weight_labels = ['YPA weight', 'AOX weight', 'KWE weight', 'VSR weight', 'VWR weight', 'ZNX weight', 'MWN weight', 'MZJ weight', 'VEO weight', 
                                'GJA weight', 'TUW weight', 'HBM weight', 'WNF weight', 'ABM weight', 'RLR weight', 'GPE weight', 'BUL weight', 'PKU weight', 
                                'VLK weight', 'VYQ weight', 'EVW weight', 'NQW weight', 'PHV weight', 'DAI weight', 'WGQ weight', 'JFL weight', 'RLF weight', 
                                'NFR weight', 'ZKY weight', 'STK weight', 'OWG weight', 'YLF weight', 'CPM weight', 'TGI weight', 'MGW weight', 'NTR weight', 
                                'KKD weight', 'JPN weight', 'CGS weight', 'VOX weight']
    

    '''
    clean_historical_data(self) - Cleans historical price data of stocks from provided Price Data excel format to backtester expected format
    
    Returns:
    Nothing, updates self.price_data and downloads cleaned price data as 'price_data_cleaned.csv' to current directory
    '''
    def clean_historical_data(self):
        # Remove first row of shares information (strategy allocates percentage of portfolio to each stock irrespective of market cap)
        self.price_data = self.price_data[1:]
        # Rename column of dates to backtester expected column name "Date"
        self.price_data = self.price_data.rename(columns={"Ticker":"Date"})
        # Convert string dates to backtester expected object type of DateTime
        self.price_data['Date'] = pd.to_datetime(self.price_data['Date'])
        # Set Date column as index of price data DataFrame
        self.price_data = self.price_data.set_index('Date')
        # Download re-formatted price data DataFrame to be passed into backtester as 'price_data_cleaned.csv'
        self.price_data.to_csv('backtesting/backtest_v2/price_data_cleaned.csv')

    # Proxy for momentum: new_var for security = (var of security - mean var)^2 + mean var
    def new_var(self, old_var):
        return (old_var - self.avg_var)**2 + self.avg_var
    '''
    historical_training(self) - On first day, initialize historical stock information variables
    Returns:
    Nothing, initializes self.returns_data, self.cov, self.mean_prices, self.vars, and self.annual_returns from historical data
    '''
    def historical_training(self):
        self.vars = self.price_data.var()
        self.cov = self.price_data.cov()
        
        # USING RETURNS PRODUCES WORSE RESULTS Convert price data to returns
        #self.returns_data = self.price_data.pct_change(1)
        #self.price_data[1:].div(self.price_data.shift(1))
        #self.returns_data = self.returns_data.dropna()
        #print(self.returns_data)

        # Calculate covariance matrix of returns between each pair of stock
        #self.vars = self.returns_data.var()
        #self.cov = self.returns_data.cov()

        # New proxy for momentum, squared difference from average market variance is sign of increasing volatility
        # find average variance of historical period
        self.avg_var = self.vars.mean()
        self.vars = self.vars.apply(self.new_var)

        # weekly average price
        weekly_avg = self.price_data.groupby(np.arange(len(self.price_data))//5).mean()

        # three month return, 50 windows one for each week, 
        # average 3 month return rate * 4 = yearly return rate
        weekly_avg_shifted = weekly_avg.shift(12)
        three_month_return = weekly_avg.sub(weekly_avg_shifted)
        three_month_return = weekly_avg.sub(weekly_avg_shifted).div(weekly_avg_shifted)
        three_month_return = three_month_return.dropna()
        mean_three_month_return = three_month_return.mean()
        self.annual_returns = mean_three_month_return.multiply(4)
        
        # Calculate mean prices of each stock
        self.mean_prices = self.price_data.mean()

        


    '''
    update(self, prices) - Update variance and projected annual returns using new prices

    :prices: - (Series) New prices for the current day with stock tickers as labels
    Returns:
    Nothing, updates self.annual_returns, self.vars, self.mean_prices, and the variances in self.cov from historical data
    '''
    def update(self, prices):
        # Calculate daily return from yesterday to today
        current_return = prices.subtract(self.last_prices).divide(self.last_prices)

        # Project today's daily return to annual return
        # Update annual returns by weighted average of today's annual return and historical annual returns
        self.annual_returns = current_return.add(249/250 * self.annual_returns)
        
        # Calculate today's variance using mean prices
        current_var = prices.subtract(self.mean_prices)
        current_var = current_var.multiply(current_var)
        # Using new var on each daily variance is not beneficial
        #self.avg_var =  (1/250 * current_var.mean()) + (249/250 * self.avg_var)
        #current_var = current_var.apply(self.new_var)

        # Update variances by weighted average of today's variances and historical variances
        self.vars = (1/250 * current_var).add(249/250 * self.vars)
        
        # Update variances in covariance matrix with new variances
        np.fill_diagonal(self.cov.values, self.vars)

        # Update mean prices with weighted average of today's prices and historical mean
        self.mean_prices = prices.multiply(1/250).add(249/250*self.mean_prices)

        # Update last prices to today's prices for tomorrow's daily return calculation
        self.last_prices = prices
        
    '''
    markowitz(self) - calculate optimal market portfolio weights given risk free rate of 0

    Returns:
    (Series) Weights of optimal market portfolio with weights labels ("<stock ticker> weights") as labels
    '''
    def markowitz(self):
        V = self.cov
        mu = np.array(self.annual_returns)

        e = np.ones(40)

        V_inv = np.linalg.inv(V)

        B = np.dot(np.dot(mu.transpose(),V_inv),e)

        W = np.dot(V_inv,mu)/B

        # Convert to backtester expected format
        weights = pd.DataFrame([W],columns=self.weight_labels)
        weights = weights.iloc[0]
        
        return weights

# Create Stocks instance to keep track of stock information starting with historical data from provided Price Data from case packet
stocks = Stocks("backtesting/backtest_v2/price_data.csv")
# Reformat price data to fit backtester expectations and download to local directory as "price_data_cleaned.csv" to pass into backtester
stocks.clean_historical_data()


'''
strat_function(preds, prices) - user specified mapping from past n days of price and analyst data to weights.
Returns: An array of asset weightings. The maximum weighting is 1, and the minimum is -1. The weights must sum to between -1 and 1. 

Refer to test datasets for the shape of input data. Both preds and prices will be 2 dimensional arrays, with number of columns equal to number of assets + 1.
Number of days equal to number of rows. The first column will be date data.

Your strategy function needs to work with this data to geenrate portfolio weights.

'''
def strat_function(preds, prices, last_weights):

    # Reformat prices as a pandas Series with stock ticker labels to match other stock information variables
    prices = pd.Series(prices[0], stocks.tickers)

    if stocks.first_run:
        # Initialize stock information on historical data
        stocks.historical_training()
        # Keep first day's prices to start calculating daily returns on the next day
        stocks.last_prices = prices
        # Calculate initial weights from historical data
        weights = stocks.markowitz()

        # Don't initialize from historical data again
        stocks.first_run = False
    else:
        # Update stock information from current prices
        stocks.update(prices)
        # Update weights
        weights = stocks.markowitz()
    print(weights.sum())
    return weights

'''
Running the backtest - starting portfolio value of 10000, reading in data from these two locations.
'''
backtest(strat_function, 10000, 'backtesting/backtest_v2/price_data_cleaned.csv', 'backtesting/backtest_v2/price_data_cleaned.csv', True, "backtesting/backtest_v2/ZWG_pm_results.csv")


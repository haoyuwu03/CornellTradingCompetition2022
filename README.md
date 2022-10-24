# CornellTradingCompetition2022
Case Packet with explanation of portfolio allocation, derivatives trading, and crypto cases:
https://docs.google.com/document/d/1o79Wrjlp7YZdR1MePZlquGTutfUfrP4gn7BsaJwEBuI/edit?usp=sharing

Code for portfolio allocation and derivatives trading in their respective folders, crypto case on apifiny.

Link to all 3 case instruction:
https://docs.google.com/document/d/1o79Wrjlp7YZdR1MePZlquGTutfUfrP4gn7BsaJwEBuI/edit#
Link to Options Case Data: 
https://drive.google.com/drive/folders/1VVCzYKQw82KPW2SflMo4jjHxY9Z6XEay
Link to PM Data: 
https://docs.google.com/spreadsheets/d/1yDqOV9RjRwrFJstonqwkJjF7Fxq87GFv0kJLWXseti8/edit#gid=0
Link to Crypto Case instructions:
https://gitlab.com/apifiny-algo/cornell-trading-competition

## Portfolio Allocation Methods:
- New estimate for historical variance using: (variance of stock - mean variance across all stocks)^2 + mean variance across all stocks
- Use four times the average of three month returns calculated from weekly average of daily returns to estimate annual returns
- Update diagonal of covariance matrix daily with new moving average variance for each stock using: (variance of new day * 1/250) + (old variance * 249/250)
- Update annual return estimates using: (new daily return * 250 * 1/250) + (old annual return estimate * 249/250)
- Inputed covariance and annual return estimates into markowitz's capital market line formula for the market portfolio weights when the risk free rate is 0 as specified in case packet: weights = (inverse covariance matrix dot product with annual return estimates) / (annual return estimates transpose multiplied with inverse covariance matrix then dot product with column of ones)

### Results of Portfolio Allocation
Stock Price Movement During Time Period

![Stocks Value Over Time](https://github.com/haoyuwu03/CornellTradingCompetition2022/blob/main/portfolio_allocation/stock_values_over_time.png)

Portfolio Value Over Time

![Portfolio Value Over Time](https://github.com/haoyuwu03/CornellTradingCompetition2022/blob/main/portfolio_allocation/Portfolio%20Value%20Over%20Time.png)

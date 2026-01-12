"""
File name: monte_carlo_MPT.py

Author: Safa Sabry

Last update: 1/12/2026

Version: 1.0

Description: This program applies Markowitz Portfolio Theory (MPT) using a Monte Carlo simulation. 
- Stocks selected: Meta Platforms Inc, Apple Inc, Microsoft Corp 
- Adjusted closing prices since 2023-01-01 are used for selected stocks using the yfinance library.
- 10000 portfolio weight combinations are generated.

Libraries used:
- yfinance
- numpy
- pandas
- matplotlib
"""

#imports
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#--- IMPORTING INFO ---  
    #ticker info 
tickers = ["META","AAPL","MSFT"]
start = "2023-01-01"
    #close price
data = yf.download(tickers, start=start, auto_adjust=True)["Close"]

#--- STOCK INFO --- 
    #daily returns
returns = data.pct_change().dropna()

    #expected annual return
mean_returns = returns.mean() * 252
print("     EXPECTED ANNUAL RETURN:")
print(mean_returns)

    #cov matrix  
cov_matrix = returns.cov()
print("\n     COVARIANCE MATRIX:")
print(cov_matrix*252)

#--- GENERATING RANDOM WEIGHTS ---  
rando = 10000
    #empty list for loop
all_weights = []
for x in range(rando):
    random_numbers = np.random.rand(len(tickers))
    weights = (random_numbers/random_numbers.sum()).tolist()
    all_weights.append(weights)

all_weights = np.array(all_weights)

#--- FINDING PORTFOLIO RISK AND RETURN RELATED TO RANDOM WEIGHTS --- 
    #empty list for risk and return
all_risk = []
all_return = []

    #portfolio risk (sd) and variance 
for x in range (rando):
    portfolio_var = all_weights[x].T @ cov_matrix @ all_weights[x]
    portfolio_risk = np.sqrt(portfolio_var)
    all_risk.append(portfolio_risk)

    #portfolio return 
for x in range(rando):
    portfolio_return = np.dot(all_weights[x], mean_returns)
    all_return.append(portfolio_return)

#--- GRAPH --- 

    #plot x, y 
for x in range(rando):
    plt.plot(all_risk[x], all_return[x], 'o', color= 'blue')

    #labels
plt.xlabel("RISK")
plt.ylabel("RETURN")
plt.title("MPT: META, AAPL, MSFT")

    #show graph 
plt.show()

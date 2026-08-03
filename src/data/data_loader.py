import numpy as np
import yfinance as yf
import pandas as pd

ticker = yf.Ticker("MSFT")
df = ticker.history(start='2020-01-01')
df.to_csv('msft_data.csv')

print('Dataframe conversion complete!')
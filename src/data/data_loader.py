import numpy as np
import yfinance as yf
import pandas as pd

ticker = yf.Ticker("MSFT")
df = ticker.history(period='max')
df.to_csv('msft_data.csv')

print('Dataframe conversion complete!')
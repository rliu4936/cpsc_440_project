import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_handler import DataHandler

# Download and prepare data
handler = DataHandler("QQQ", start_date="2021-01-01", 
                      end_date=datetime.today().strftime("%Y-%m-%d"))
df = handler.download_data()

print(df)
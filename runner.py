import yfinance as yf
from Oracle import stock_oracle
def runner():
    print("hello, this recurrent neural network is a next day (at close) stock price predictor")
    tick = input("enter the stock ticker that you'd like to predict(TSLA, AAPL, AMZN ETC.)")
    if not is_valid_ticker(tick):
        print(f"{tick} is not a valid ticker.")
    else:
        stock_oracle(tick)

def is_valid_ticker(ticker):
    try:
        yf.Ticker(ticker).info
        return True
    except ValueError:
        return False
runner()

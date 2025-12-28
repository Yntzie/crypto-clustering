import yfinance as yf

def load_crypto_data(period="1y"):
    symbols = [
        "BTC-USD","ETH-USD","BNB-USD","SOL-USD",
        "ADA-USD","XRP-USD","DOGE-USD","AVAX-USD"
    ]

    data = {}
    for s in symbols:
        df = yf.download(s, period=period, interval="1d", progress=False)
        data[s] = df

    return data

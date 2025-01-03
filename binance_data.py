import time
import aiohttp
import pandas as pd
import asyncio

BINANCE_API = "https://api.binance.com/api/v3"


async def get_tokens():
    session = aiohttp.ClientSession()
    url = f'{BINANCE_API}/exchangeInfo'

    response = await session.get(url)
    data = await response.json()

    usdt_pairs = [symbol['symbol'] for symbol in data['symbols']
                  if 'USDT' == symbol['symbol'][-4:]]

    await session.close()

    return usdt_pairs


async def get_klines(session: aiohttp.ClientSession, symbol, limit,
                     interval, startTime):
    result_data = []
    url = f"{BINANCE_API}/klines"
    params = {
        "startTime": startTime,
        "limit": limit,
        "interval": interval,
        "symbol": symbol
    } if startTime is not None else {
        "limit": limit,
        "interval": interval,
        "symbol": symbol
    }

    response = await session.get(url, params=params)
    response_data = await response.json()

    for kline in response_data:
        result_data.append({
                "Timestamp": pd.to_datetime(float(kline[0]), unit="ms"),
                "Open": float(kline[1]),
                "Close": float(kline[4]),
                "High": float(kline[2]),
                "Low": float(kline[3]),
                "Volume": float(kline[5]),
        })

    last_timestamp = response_data[-1][0]
    return pd.DataFrame(result_data), last_timestamp+1


def get_techincal_information(dataframe: pd.DataFrame, group_id):
    close = dataframe["Close"]
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    dataframe['RSI'] = 100 - (100 / (1 + rs))
    dataframe['EMA_26'] = close.ewm(span=26, adjust=False).mean()
    dataframe['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    dataframe['MACD'] = dataframe['EMA_12'] - dataframe['EMA_26']
    dataframe['SMA_12'] = close.rolling(window=12).mean()
    dataframe['SMA_26'] = close.rolling(window=26).mean()
    dataframe['Signal Line'] = dataframe['MACD'].ewm(span=9,
                                                     adjust=False).mean()
    dataframe['MACD Histogram'] = dataframe['MACD'] - dataframe['Signal Line']
    dataframe['Upper_Bridge'] = dataframe['SMA_26'] + (2 * close.rolling(
        window=26).std())
    dataframe['Lower_Bridge'] = dataframe['SMA_26'] - (2 * close.rolling(
        window=26).std())
    dataframe['L_n'] = dataframe['Low'].rolling(window=14).min()
    dataframe['H_n'] = dataframe['High'].rolling(window=14).max()
    dataframe['K'] = ((close - dataframe['L_n']) /
                      (dataframe['H_n'] - dataframe['L_n'])) * 100
    dataframe['D'] = dataframe['K'].rolling(window=3).mean()
    dataframe['Prev_Close'] = close.shift(1)
    dataframe['TR'] = dataframe[['High', 'Low', 'Prev_Close']].max(axis=1) - dataframe[['High', 'Low', 'Prev_Close']].min(axis=1)
    dataframe['A/D'] = ((close - dataframe['Low']) -
                        (dataframe['High'] - close)) / dataframe['TR'] * dataframe['Volume']
    dataframe['A/D'] = dataframe['A/D'].cumsum()

    dataframe['ATR'] = dataframe['TR'].rolling(window=14).mean()
    dataframe['Next_Close'] = close.shift(-1)
    dataframe['group_id'] = group_id
    dataframe = dataframe.dropna()
    return dataframe


async def get_klines_for_train(symbol, group_id=0):
    session = aiohttp.ClientSession()

    prices_data = pd.DataFrame()

    last_timestamp = 0
    current_time = time.time() * 1000

    while abs(last_timestamp - current_time) > 3600 * 1000:
        try:
            new_data, last_timestamp = await get_klines(session, symbol, 1000,
                                                        "1h", last_timestamp)

            prices_data = pd.concat([prices_data, pd.DataFrame(new_data)],
                                    ignore_index=True)
        except Exception:
            await session.close()
            break

    await session.close()
    prices_data = get_techincal_information(prices_data, group_id)
    return prices_data


async def main():
    symbols = ["XRPUSDT", "SOLUSDT"]
    tasks = [get_klines_for_train(symbols[i], i)
             for i in range(len(symbols))]
    j = 0
    for i in await asyncio.gather(*tasks):
        i.to_csv(f"{symbols[j]}.csv")
        j+=1

if __name__ == "__main__":
    asyncio.run(main())

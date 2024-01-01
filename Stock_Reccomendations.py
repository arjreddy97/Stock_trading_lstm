import trainModels
import LSTM_model
from LSTM_model import Model
from trainModels import getSymbols,fetch_data,fetch_trades,fetch_quotes
import asyncio
import aiohttp
import aiolimiter
from aiolimiter import AsyncLimiter
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.requests import StockTradesRequest
import calendar
import datetime
from datetime import timezone
from datetime import datetime, timedelta
import csv



async def getSignals(year,month,day,start_hour,start_min,end_hour,end_min):

    rate_limit = 9500
    limiter = AsyncLimiter(rate_limit, 60)


    potential_gainers = []
    potential_losers = []
    delta_loss = []
    delta_gain = []
    symbols = getSymbols()


    start_time = datetime(year, month, day, start_hour, start_min)  # Example start time
    end_time = datetime(year, month, day, end_hour, end_min)

    for symbol in symbols:


        current_time = start_time

        tasks = []
        data = []

        while current_time <= end_time:

            current_time2 = current_time
            current_time2 += timedelta(minutes=1)

            task = fetch_data(current_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),current_time2.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),symbol,data,current_time,limiter)
            tasks.append(task)
            current_time = current_time2

        await asyncio.gather(*tasks)


        data.sort(key = lambda x: x[6])

        current_data = []
        for ele in data:
            if ele[0] == None or ele[1] == None or ele[2] == None or ele[3] == None or ele[4] == None or ele[5] == None or ele[6] == None:
                continue
            #print(ele)
            current_data.append(ele[0:6])
        print(data)
        print(symbol)
        print(len(current_data))
        if len(current_data) < 15:

            print("Not enough data from today ",symbol)
            continue

        lastIdx = len(current_data) - 1
        input_data =  current_data[lastIdx - 14:len(current_data)]

        model = Model(symbol)

        predictions = model.predict(input_data)
        print(symbol+" ",predictions)

        starting_price = current_data[0][4]
        closing_price = predictions[-1][4]

        print("starting price ", starting_price)
        print("closing price ", closing_price)

        percentage_change = ((closing_price - starting_price) / starting_price ) * 100
        delta = closing_price - starting_price
        if delta >= 0:
            delta_gain.append((symbol,delta,closing_price,starting_price))
        else:
            delta_loss.append((symbol,delta,closing_price,starting_price))
        if percentage_change >= 0:

            potential_gainers.append((symbol,percentage_change))

        else:

            potential_losers.append((symbol,percentage_change))


    potential_gainers.sort(key = lambda x: x[1])
    potential_gainers = potential_gainers[::-1]
    potential_gainers.sort(key = lambda x: x[1])
    print(" Percentage gainers",potential_gainers)
    print(" Percentage losers",potential_losers)
    delta_gain.sort(key = lambda x: x[1])
    delta_gain = delta_gain[::-1]
    delta_loss.sort(key = lambda x: x[1])
    print(" Delta gainers",delta_gain)
    print(" Delta loss",delta_loss)

    return

if __name__ == "__main__":
    asyncio.run(getSignals(2023,10,17,9,30,10,5))

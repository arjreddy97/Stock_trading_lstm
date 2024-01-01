from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.requests import StockTradesRequest
import calendar
import datetime
from datetime import timezone
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import csv
import LSTM_model
from LSTM_model import Model
import asyncio
import aiohttp
import aiolimiter
from aiolimiter import AsyncLimiter

KEY = 'PKHEEY96SUMFXLJ945C0'
SECRET = 'lOysAxsfdGXjjZSJPEarfvUtiAnj13NGqHkYIDws'



def getSymbols():
    file_name = 'sp-500-index-08-21-2023.csv'
    symbols = []
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
    # Skip the header row
        header = next(csvreader)

        for row in csvreader:
            symbol = row[0]
            name = row[1]
            symbols.append(symbol)

    return symbols

def fetch_and_process_data(current_time,current_time2, symbol, client,data):

    request_params_quote = StockQuotesRequest(symbol_or_symbols=symbol,start = current_time,end = current_time2, limit = 1)
    request_param_trade = StockTradesRequest(symbol_or_symbols=symbol,start = current_time,end = current_time2,limit = 1)
    try:
        quote_object = client.get_stock_quotes(request_params_quote)
        trade_object = client.get_stock_trades(request_param_trade)
        #print(quote)
        quote_object_data = quote_object[symbol][0]
        trade_object_data = trade_object[symbol][0]

        transaction = [quote_object_data.ask_price,quote_object_data.ask_size,
        quote_object_data.bid_price,quote_object_data.bid_size,trade_object_data.price,trade_object_data.size]

        transaction.append(current_time)
        data.append(transaction)
        print(transaction,quote_object_data.timestamp)


    #except AttributeError:
    #    print("None Error")

    except Exception as e:
        # Print the exception
        print(f"An error occurred: {e}")


async def fetch_quotes(base_url,symbol,quote_query,transaction,limiter):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        endpoint = f"/stocks/{symbol}/quotes"
        url = f"{base_url}{endpoint}"
        headers = {
            "APCA-API-KEY-ID": KEY,
            "APCA-API-SECRET-KEY": SECRET,
        }

        async with limiter:
            async with session.get(url, headers=headers, params=quote_query) as response:
                if response.status == 200:
                    quote_object = await response.json()

                    if quote_object["quotes"] == None:
                        return
                        #print(quote_object)
                    quote_object_data = quote_object["quotes"][0]
                    # Process the quotes data as needed
                    transaction[0] = float(quote_object_data["ap"])
                    transaction[1] = float(quote_object_data["as"])
                    transaction[2] = float(quote_object_data["bp"])
                    transaction[3] = float(quote_object_data["bs"])

                    #print("Quotes Data:")
                    #print(quote_object_data)
                else:
                    print(f"Quotes Error: {response.status} - {await response.text()}")

    return


async def fetch_trades(base_url,symbol,trade_query,transaction,limiter):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        endpoint = f"/stocks/{symbol}/trades"
        url = f"{base_url}{endpoint}"
        headers = {
            "APCA-API-KEY-ID": KEY,
            "APCA-API-SECRET-KEY": SECRET,
        }

        async with limiter:
            async with session.get(url, headers=headers, params=trade_query) as response:
                if response.status == 200:

                    trade_object = await response.json()


                    if trade_object["trades"] == None:
                        return

                        #print(trade_object)
                    trade_object_data = trade_object["trades"][0]
                    # Process the trades data as neede
                    transaction[4] = float(trade_object_data["p"])
                    transaction[5] = float(trade_object_data["s"])

                    #print("Trades Data:")
                    #print(trade_object_data)
                else:
                    print(f"Trades Error: {response.status} - {await response.text()}")
    return



async def fetch_data(start_time,end_time,symbol,data,current_time,limiter):

    retries = 0
    max_retries = 100
    while retries < max_retries:
        try:
            base_url = "https://data.alpaca.markets/v2"


            query_parameters = {
            "start": start_time,  # Replace with your desired start time in RFC-3339 format
            "end": end_time,    # Replace with your desired end time in RFC-3339 format
            "limit": 1,  # Replace with your desired limit (1-10000) # Replace with "iex" or "sip"# Replace with a pagination token if needed
            "currency": "USD",  # Replace with the desired currency
            }
            transaction = [None] * 7
            await asyncio.gather(fetch_quotes(base_url,symbol,query_parameters,transaction,limiter),fetch_trades(base_url,symbol,query_parameters,transaction,limiter))
            transaction[6] = current_time
            #print("Transaction ",transaction,symbol)
            data.append(transaction)
            return

        except Exception as e:
            print(f"Attempt {retries + 1} failed: {e}")
            retries += 1

    return



async def trainModels(startYear,startMonth,endYear,endMonth):
    client = StockHistoricalDataClient(KEY, SECRET)
    symbols = getSymbols()
    rate_limit = 9500
    limiter = AsyncLimiter(rate_limit, 60)
    #banned = set(["A","AAL","AAP","AAPL","ABBV","ABC","ABT","ACGL","ACN","ADBE"])
    #banned_count = 1
    #banned = 502
    for symbol in symbols:
        #if symbol in banned:
            #continue
        #if banned_count <= banned:
            #banned_count += 1
            #continue
        start_date = datetime(startYear, startMonth, 1)
        end_date = datetime(endYear, endMonth, 1)
        current_date = start_date
        while current_date <= end_date:

            current_year,current_month = current_date.year,current_date.month
            # Increment the month

            #for month in range(startMonth,endMonth+1)

            _,num_days = calendar.monthrange(current_year,current_month)


            for day in range(1, num_days + 1):

                day_of_week = calendar.weekday(current_year, current_month, day)

                if day_of_week >= 5:
                    continue

                if day != 16:
                    continue

                data = []

                start_time = datetime(current_year, current_month, day, 9, 30)  # Example start time
                end_time = datetime(current_year, current_month, day, 20, 0)


                tasks = []
                current_time = start_time
                while current_time <= end_time:


                    current_time2 = current_time
                    current_time2 += timedelta(minutes=1)

                    task = fetch_data(current_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),current_time2.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),symbol,data,current_time,limiter)
                    tasks.append(task)
                    current_time = current_time2

                await asyncio.gather(*tasks)


                data.sort(key = lambda x: x[6])

                training_data = []
                for ele in data:
                    if ele[0] == None or ele[1] == None or ele[2] == None or ele[3] == None or ele[4] == None or ele[5] == None or ele[6] == None:
                        continue
                    print(ele)
                    training_data.append(ele[0:6])

                    #print(training_data,type(training_data[-1][-2]))

                print("length of training data " ,len(training_data))
                try:
                        #print(training_data,len(training_data))
                    model = Model(symbol)
                    model.trainModel(training_data)
                    X_test = model.X_test
                    if len(X_test):
                        print("predictions ",model.predict(X_test[0]))
                        print("actual Y_test", model.scaler.inverse_transform(model.Y_test[0]))
                        #print("actual Y_test", model.Y_test)
                except Exception as e:
                    # Print the exception
                    print(f"An error occurred: {e}")


            current_date += relativedelta(months=1)

        log_file = open('log_file.txt', 'a')
        log_file.write("Stock "+symbol+" has been trained ")#+str(current_month)+"/"+str(current_year)+ '\n')
        log_file.close()


if __name__ == "__main__":
    asyncio.run(trainModels(2023,10,2023,10))

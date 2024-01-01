import csv
import LSTM_model
from LSTM_model import Model
import json
from keras.models import model_from_json
from keras.models import load_model
import pickle
import pymongo
from pymongo import MongoClient
import certifi


file_name = 'sp-500-index-08-21-2023.csv'






def getSymbols():
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


def initialize_models(symbols):

    for symbol in symbols:

        model = Model(symbol)
        model.createModel()


    return



symbols = getSymbols()
initialize_models(symbols)


"""



uri = "mongodb+srv://arjunreddy31:FOP2RXvdOeyt7Bt3@cluster0.43vp6jy.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(uri,tlsCAFile=certifi.where())  # Connect to your MongoDB instance
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)



db = client["StockTrading"]
collection = db["models"]
model_doc = collection.find_one({"symbol": 'AAPL'})
if model_doc == None:
    print("model doc == None")

"""

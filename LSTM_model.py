import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import json
from keras.models import model_from_json
from keras.models import load_model
import pickle
import pymongo
from pymongo import MongoClient
import certifi
import os
import subprocess
import warnings
import joblib






class Model:

    def __init__(self,symbol):
        self.symbol = symbol
        #self.uri = "mongodb+srv://arjunreddy31:FOP2RXvdOeyt7Bt3@cluster0.43vp6jy.mongodb.net/?retryWrites=true&w=majority"
        #self.client = pymongo.MongoClient(self.uri,tlsCAFile=certifi.where())  # Connect to your MongoDB instance
        #self.db = self.client["StockTrading"]
        #self.collection = self.db["models"]
        self.input_sequence_length = 15
        self.output_sequence_length = 360
        self.n_features = 6
        self.X_test = None
        self.Y_test = None
        self.model = None
        self.scaler = None
        self.loadModel(self.symbol)
        self.not_enough_data_count = 0


    def trainModel(self,data):

        #self.scaler = MinMaxScaler()

        #scaled_data = self.scaler.fit_transform(data)
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data)


        input_sequence_length = self.input_sequence_length
        output_sequence_length = self.output_sequence_length  # Number of minutes in the next 7 hours

        X = []
        y = []

        if len(scaled_data) < 374:
            print("Not enough data 1")
            log_file = open('log_file.txt', 'a')
            log_file.write("Stock "+self.symbol+" not enough data "+'\n')
            log_file.close()
            self.not_enough_data_count += 1

            return
        iterations = len(scaled_data) - output_sequence_length
        print("iterations ",iterations)
        print("len_scaled_data", len(scaled_data))
        for i in range(len(scaled_data)  - output_sequence_length):

            if (i + input_sequence_length > len(scaled_data) or (i + input_sequence_length + output_sequence_length > len(scaled_data))):
                break

            X.append(scaled_data[i:i+input_sequence_length])
            y.append(scaled_data[i+input_sequence_length:i+input_sequence_length+output_sequence_length])

        X = np.array(X)
        y = np.array(y)

        print()

        if X.shape[0] < 1 or y.shape[0] < 1 or X.shape[0] != y.shape[0]:

            print("Not enough data 2")
            log_file = open('log_file.txt', 'a')
            log_file.write("Stock "+self.symbol+" not enough data "+'\n')
            log_file.close()

            self.not_enough_data_count += 1

            return
        print("X shape ",X.shape)
        print("y shape ",y.shape)

        # Split data into training and testing sets
        train_size = int(0.8 * len(X))
        if train_size <= 0:
            print("Not enough data 3")
            return
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        y_train_reshaped = y_train.reshape(-1, output_sequence_length * self.n_features)

        self.X_test = X_test
        self.Y_test = y_test

        print(self.symbol,"Model training beginning")

        # Train the model
        self.model.fit(X_train, y_train_reshaped, epochs=50, batch_size=32)

        print(self.symbol, "Model trained")

        self.saveModel(self.symbol)
        self.saveScaler()

        return




    def createModel(self):

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(self.input_sequence_length, self.n_features)))
        model.add(Dense(self.output_sequence_length * self.n_features,activation='linear'))

        model.compile(optimizer='adam', loss='mse')

        self.model = model

        print(self.symbol, " Model Created")

        self.createModelFile(self.symbol)
        self.createScalerFile()

        self.saveModel(self.symbol)


        return self.model

    def predict(self,data):
        ## un comment this when doing actual predictions
        """
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        data1 = np.array(scaled_data)
        input_data = data1.reshape(1, self.input_sequence_length, self.n_features)
        predicted_data = self.model.predict(input_data)
        predicted_data = predicted_data.reshape(-1,self.n_features)
        predicted_data_original = scaler.inverse_transform(predicted_data)
        return predicted_data_original
        """


        self.loadScaler()
        #self.scaler = StandardScaler()
        #scaled_data = self.scaler.fit_transform(data)
        scaled_data = self.scaler.transform(data)
        #data1 = np.array(data)
        data1 = np.array(scaled_data)
        input_data = data1.reshape(1, self.input_sequence_length, self.n_features)
        predicted_data = self.model.predict(input_data)

        # Inverse transform to get predictions in original scale
        predicted_data = predicted_data.reshape(-1,self.n_features)
        predicted_data_original = self.scaler.inverse_transform(predicted_data)

        return predicted_data_original



    def createModelFile(self,symbol):
        folder_path = "/Users/larjarj/Desktop/StockTrading/ModelFolder"
        file_name = symbol+".h5"
        file_path = os.path.join(folder_path, file_name)
        subprocess.run(["touch", file_path])
        print(f"File '{file_name}' created in '{folder_path}'.")
        return

    def saveModel(self,symbol):
        warnings.filterwarnings("ignore", message="You are saving your model as an HDF5 file")
        self.model.save("/Users/larjarj/Desktop/StockTrading/ModelFolder/"+symbol+".h5")
        print(self.symbol," Model Saved")

        return

    def saveScaler(self):
        scaler_path = "/Users/larjarj/Desktop/StockTrading/ModelFolder/"+self.symbol+"_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(self.symbol," Scaler Saved")
        return


    def createScalerFile(self):
        scaler_filename = self.symbol+"_scaler.pkl"
        folder_path = "/Users/larjarj/Desktop/StockTrading/ModelFolder"
        file_path = os.path.join(folder_path, scaler_filename)
        subprocess.run(["touch", file_path])
        print(f"File '{scaler_filename}' created in '{folder_path}'.")
        return



    def loadModel(self,symbol):
        model_path = "/Users/larjarj/Desktop/StockTrading/ModelFolder/"+symbol+".h5"
        if os.path.exists(model_path):
            loaded_model = load_model(model_path)
            self.model = loaded_model
            print(self.symbol," Model Loaded")

        else:
            print(f"Model file '{model_path}' not found")

        return


    def loadScaler(self):
        model_path = "/Users/larjarj/Desktop/StockTrading/ModelFolder/"+self.symbol+"_scaler.pkl"
        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)
            self.scaler = loaded_model
            print(self.symbol," Scaler Loaded")

        else:
            print(f"Model file '{model_path}' not found")


        return






""""
    def saveModelMongo(self):


        #self.model.save('model_save_file.h5')
        serialized_model = pickle.dumps(self.model)
        scaler_bytes = pickle.dumps(self.scaler)

        self.collection.update_one(
        filter={"symbol": self.symbol},  # Replace with the appropriate identifier for the scaler item
        update={"$set": {"model_data": serialized_model,"scaler_data":scaler_bytes}},
        upsert=True)







        print(self.symbol, " Model saved")

        return


    def loadModelMongo(self):

        model_doc = self.collection.find_one({"symbol": self.symbol})
        if model_doc == None:
            print(self.symbol," Model not able to load")
            return None,None
        print(self.symbol, "model_loaded")

        serialized_model = model_doc["model_data"]
        serialized_scaler = model_doc["scaler_data"]

        loaded_model =  pickle.loads(serialized_model)
        loaded_scaler = pickle.loads(serialized_scaler)
        self.model = loaded_model
        self.scaler = loaded_scaler

        return self.model,self.scaler

"""

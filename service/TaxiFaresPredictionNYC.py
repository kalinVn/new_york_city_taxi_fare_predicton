import numpy as np
import pandas as pd
import config
import sys
import tensorflow as tf

from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from service.Preprocess import Preprocess
from service.FeatureEngineer import FeatureEngineer


class TaxiFaresPredictionNYC:

    def __init__(self):
        self.df = None
        self.x = None
        self.y = None
        self.x_train = None
        self.y_test = None
        self.x_test = None
        self.y_train = None
        self.df_prescaled = None
        self.f_engineer = None
        self.model = Sequential()
        self.preprocessObj = Preprocess()

    def feature_engineer(self):
        self.f_engineer = FeatureEngineer(self.df)
        self.f_engineer.create_date_columns()
        self.f_engineer.create_dist_column()
        self.f_engineer.create_airport_dist_features()

    def preprocess(self):
        self.preprocessObj.remove_missing_values()
        self.preprocessObj.remove_fare_amount_outliers()
        self.preprocessObj.replace_passenger_count_outliers()
        self.preprocessObj.remove_lat_long_outliers()
        self.df = self.preprocessObj.get_dataset()

        self.feature_engineer()

        self.df = self.f_engineer.get_dataset()

        self.df_prescaled = self.df.copy()

        self.df = self.preprocessObj.scale()

        self.x = self.df.loc[:, self.df.columns != 'fare_amount']
        self.y = self.df.fare_amount

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2)

    def create(self):
        self.model.add(Dense(128, activation='relu', input_dim=self.x_train.shape[1]))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1))

    def compile(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model.fit(self.x_train, self.y_train, epochs=1)

    def predict(self):
        train_predict = self.model.predict(self.x_train)
        train_mrs_error = np.sqrt(mean_squared_error(self.y_train, train_predict))
        print("Train RMSE: {:0.2f}".format(train_mrs_error))

        test_predict = self.model.predict(self.x_test)
        test_mrs_error = np.sqrt(mean_squared_error(self.y_test, test_predict))
        print("Test RMSE: {:0.2f}".format(test_mrs_error))

    def get_dataset(self):
        return self.df





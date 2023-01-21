import pandas as pd
import config
from sklearn.preprocessing import scale


class Preprocess:

    def __init__(self):
        csv_path = config.CSV_PATH
        self.df = pd.read_csv(csv_path, parse_dates=['pickup_datetime'], nrows=300000)
        self.df_scaled = None

    def remove_lat_long_outliers(self):
        nyc_min_longitude = -74.05
        nyc_max_longitude = -73.75

        nyc_min_latitude = 40.63
        nyc_max_latitude = 40.85

        for long in ['pickup_longitude', 'dropoff_longitude']:
            self.df = self.df[(self.df[long] > nyc_min_longitude) & (self.df[long] < nyc_max_longitude)]

        for lat in ['pickup_latitude', 'dropoff_latitude']:
            self.df = self.df[(self.df[lat] > nyc_min_latitude) & (self.df[lat] < nyc_max_latitude)]

    def replace_passenger_count_outliers(self):
        mode = self.df['passenger_count'].mode()
        self.df.loc[self.df['passenger_count'] == 0, 'passenger_count'] = mode

    def remove_fare_amount_outliers(self):
        self.df = self.df[(self.df['fare_amount'] >= 0) & (self.df['fare_amount'] <= 100)]

    def remove_missing_values(self):
        self.df.dropna()

    def get_dataset(self):
        return self.df

    def scale(self):
        self.df = self.df.drop(['key'], axis=1)
        self.df = self.df.drop(['pickup_datetime'], axis=1)
        self.df_scaled = self.df.drop(['fare_amount'], axis=1)

        self.df_scaled = scale(self.df_scaled)
        cols = self.df.columns.tolist()
        cols.remove('fare_amount')

        self.df_scaled = pd.DataFrame(self.df_scaled, columns=cols, index=self.df.index)
        self.df_scaled = pd.concat([self.df_scaled, self.df['fare_amount']], axis=1)
        self.df = self.df_scaled.copy()

        return self.df



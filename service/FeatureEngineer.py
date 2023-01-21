from utils import euc_distance


class FeatureEngineer:

    def __init__(self, df):
        self.df = df

    def create_dist_column(self):
        self.df['distance'] = euc_distance(self.df['pickup_latitude'],
                                           self.df['pickup_longitude'],
                                           self.df['dropoff_latitude'],
                                           self.df['dropoff_longitude'])

    def get_dataset(self):
        return self.df

    def create_date_columns(self):
        self.df['year'] = self.df['pickup_datetime'].dt.year
        self.df['month'] = self.df['pickup_datetime'].dt.year
        self.df['day'] = self.df['pickup_datetime'].dt.day
        self.df['day_of_week'] = self.df['pickup_datetime'].dt.dayofweek
        self.df['hour'] = self.df['pickup_datetime'].dt.hour
        self.df.drop(['pickup_datetime'], axis=1)

    def create_airport_dist_features(self):
        airports = {
            'JFK_Airport': (-73.78, 40.643),
            'Laguardia_Airport': (-73.87, 40.77),
            'Newark_Airport': (-74.18, 40.69)
        }

        for k in airports:

            euc_dist_pickup = euc_distance(self.df['pickup_latitude'], self.df['pickup_longitude'], airports[k][1],
                                           airports[k][0])
            self.df['pickup_dist_' + k] = euc_dist_pickup

            euc_dist_dropoff = euc_distance(self.df['dropoff_latitude'], self.df['dropoff_longitude'], airports[k][1],
                                            airports[k][0])
            self.df['dropoff_dist_' + k] = euc_dist_dropoff




import numpy as np
from factory.Service import Service
from visualizator import hist_plot, scatter_plot

def taxi_fares_prediction():
    factory_service = Service()
    service = factory_service.get_service('NN')
    service.preprocess()
    service.create()
    service.compile()
    service.predict()

    # df = service.get_dataset()
    # visualize(df)


def visualize(df):
    params = dict(bins=np.arange(8) - 0.5, ec='black')
    xlabel = 'Day of week (0=Monday, 6=Sunday)'
    title = 'Day of week Histogram'
    # hist_plot(column=df['day_of_week'], params=params, xlabel=xlabel, title=title)

    xlabel = 'Hour'
    title = 'Pick Hour Histogram'
    params = dict(bins=24, ec='black')
    # hist_plot(column=df['hour'], params=params, xlabel=xlabel, title=title)

    xlabel = 'Fare'
    title = 'Histogram of Fares'
    params = dict(bins=500)
    # hist_plot(column=df['fare_amount'],  params=params, xlabel=xlabel, title=title)

    xlabel = 'Passenger Count'
    title = 'Histogram of Passenger Count'
    params = dict(bins=6, ec="black")
    # hist_plot(column=df['passenger_count'], params=params, xlabel=xlabel, title=title)

    # params = ('pickup_longitude', 'pickup_latitude')
    params = ('fare_amount', 'distance')
    scatter_plot(df, params)


taxi_fares_prediction()


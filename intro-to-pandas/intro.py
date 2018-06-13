import numpy as np
import pandas as pd

# print(pd.__version__)


def small_data():
    # city_names = pd.Series(['Cambridge', 'London', 'Chester'])
    # population = pd.Series([123123, 10456234, 65789])
    city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
    population = pd.Series([852469, 1015785, 485199])
    cities = pd.DataFrame({'City Name': city_names, 'Population': population})
    """
    print(cities)
    print(type(cities['City Name']))
    print(cities['City Name'])
    print(type(cities['City Name'][1]))
    print(cities['City Name'][1])
    print(type(cities[0:2]))
    print(cities[0:2])
    print(population / 1000)
    print(np.log(population))
    print(population.apply(lambda x: x > 1000000))
    """
    cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
    cities['Population density'] = cities['Population'] / cities['Area square miles']
    saint = city_names.apply(lambda x: x.startswith('San'))
    large_area = cities['Area square miles'] > 50
    cities['large saint'] = saint & large_area
    # or from tutorial
    # cities['Is wide and has saint name'] = (
    #     (cities['Area square miles'] > 50) &
    #     cities['City name'].apply(lambda name: name.startswith('San')
    # )
    # print(cities['large saint'])
    # print(city_names.index)
    # print(cities.index)
    cities.reindex([2, 0, 1])
    cities.reindex(np.random.permutation(cities.index))
    print(cities)


def big_data():
    california_housing_dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
        sep=","
    )
    print(california_housing_dataframe.describe())
    print(california_housing_dataframe.head())
    # california_housing_dataframe.hist('housing_median_age')


small_data()

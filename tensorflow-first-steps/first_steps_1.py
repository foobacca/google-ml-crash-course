import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    Trains a linear regression model of one feature

    Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: size of batches to be passed to the model
        shuffle: whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated.  None is
                    repeat indefinitely.
    Return:
        Tuple of (features, labels) for next data batch
    """
    # convert pandas data into dict of numpy arrays
    features = {key: np.array(value) for key, value in dict(features).items()}

    # construct a dataset and configure batch/repeating
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def create_dataset():
    california_housing_dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
        sep=","
    )

    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index)
    )
    california_housing_dataframe["median_house_value"] /= 1000.0
    # print(california_housing_dataframe)
    # print(california_housing_dataframe.describe())
    return california_housing_dataframe


def dataset_stats(california_housing_dataframe):
    min_house_value = california_housing_dataframe["median_house_value"].min()
    max_house_value = california_housing_dataframe["median_house_value"].max()
    min_max_difference = max_house_value - min_house_value
    print("Min. Median House Value: {0:0.3f}".format(min_house_value))
    print("Max. Median House Value: {0:0.3f}".format(max_house_value))
    print("Difference between Min. and Max.: {0:0.3f}".format(min_max_difference))


def plot_sample_predict(california_housing_dataframe, linear_regressor):
    sample = california_housing_dataframe.sample(n=300)
    x0 = sample['total_rooms'].min()
    x1 = sample['total_rooms'].max()
    weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
    y0 = weight * x0 + bias
    y1 = weight * x1 + bias
    plt.plot([x0, x1], [y0, y1], c='r')
    plt.ylabel('median_house_value')
    plt.xlabel('total_rooms')
    plt.scatter(sample['total_rooms'], sample['median_house_value'])
    plt.show()


def predict_stats(california_housing_dataframe, predictions, targets):
    # print mean squared error and root mean squared error
    mean_squared_error = metrics.mean_squared_error(predictions, targets)
    root_mean_squared_error = math.sqrt(mean_squared_error)
    print('MSE (on training data): {0:.3f}'.format(mean_squared_error))
    print('RMSE (on training data): {0:.3f}'.format(root_mean_squared_error))
    dataset_stats(california_housing_dataframe)

    callibration_data = pd.DataFrame()
    callibration_data['predictions'] = pd.Series(predictions)
    callibration_data['targets'] = pd.Series(targets)
    print(callibration_data.describe())


def main():
    california_housing_dataframe = create_dataset()
    my_feature = california_housing_dataframe[["total_rooms"]]
    feature_columns = [tf.feature_column.numeric_column("total_rooms")]
    targets = california_housing_dataframe["median_house_value"]

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer,
    )

    linear_regressor.train(
        input_fn=lambda: my_input_fn(my_feature, targets),
        steps=100
    )

    # create input fn for prediction
    prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
    # call predict()
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    # convert to NumPy so we can calc error metrics
    predictions = np.array([item['predictions'][0] for item in predictions])

    predict_stats(california_housing_dataframe, predictions, targets)
    plot_sample_predict(california_housing_dataframe, linear_regressor)


if __name__ == '__main__':
    main()

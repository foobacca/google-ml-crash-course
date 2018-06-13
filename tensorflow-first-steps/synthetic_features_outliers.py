import math

from IPython import display
from matplotlib import cm
# from matplotlib import gridspec
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
    dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
        sep=","
    )
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    dataframe["median_house_value"] /= 1000.0
    dataframe["rooms_per_person"] = dataframe["total_rooms"] / dataframe["population"]
    # print(dataframe)
    # print(dataframe.describe())
    return dataframe


def get_linear_regressor(learning_rate, feature_name):
    feature_columns = [tf.feature_column.numeric_column(feature_name)]
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    return tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer,
    )


class Plotter(object):

    def __init__(self, label, feature, periods, sample):
        self.label = label
        self.feature = feature
        self.sample = sample
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.title("Learned line by period")
        plt.ylabel(self.label)
        plt.xlabel(self.feature)
        plt.scatter(self.sample[feature], self.sample[label])
        self.colours = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    def weight_bias(self, linear_regressor):
        weights_variable = 'linear/linear_model/{}/weights'.format(self.feature)
        weight = linear_regressor.get_variable_value(weights_variable)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
        return weight, bias

    def add_period_line(self, linear_regressor, period):
        weight, bias = self.weight_bias(linear_regressor)
        initial_y_extents = np.array([0, self.sample[self.label].max()])
        x_extents = (initial_y_extents - bias) / weight
        x_extents = np.maximum(
            np.minimum(x_extents, self.sample[self.feature].max()),
            self.sample[self.feature].min()
        )
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=self.colours[period])

    def add_loss_metrics_plot(self, root_mean_squared_errors):
        # graph loss metrics over period
        plt.subplot(1, 2, 2)
        plt.title("RMSE vs periods")
        plt.ylabel('RMSE')
        plt.xlabel('periods')
        plt.tight_layout()
        plt.plot(root_mean_squared_errors)

    def show(self):
        plt.show()


def calc_rmse(np_predictions, targets):
    # convert to NumPy so we can calc error metrics
    return math.sqrt(metrics.mean_squared_error(np_predictions, targets))


def gen_callibration_data(np_predictions, targets):
    # table with callibration data
    callibration_data = pd.DataFrame()
    callibration_data['predictions'] = pd.Series(np_predictions)
    callibration_data['targets'] = pd.Series(targets)
    return callibration_data


def train_model(learning_rate, steps, batch_size, input_feature):
    """
    Trains a linear regression model of one feature.

    Args:
        learning_rate: `float`, the learning rate.
        steps: non-zero `int`, total number of training steps. A training step
               consists of a forward and backward pass using a single batch.
        batch_size: non-zero `int`, the batch size.
        input_feature: `str` specifying the column to use as an input feature
    """
    periods = 10
    steps_per_period = steps // periods

    california_housing_dataframe = create_dataset()
    feature_data = california_housing_dataframe[[input_feature]].astype('float32')
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label].astype('float32')

    train_input_fn = lambda: my_input_fn(feature_data, targets, batch_size=batch_size)
    predict_input_fn = lambda: my_input_fn(feature_data, targets, num_epochs=1, shuffle=False)
    linear_regressor = get_linear_regressor(learning_rate, input_feature)

    sample = california_housing_dataframe.sample(n=300)
    plotter = Plotter(my_label, input_feature, periods, sample)

    print('Training model')
    root_mean_squared_errors = []
    for period in range(periods):
        linear_regressor.train(input_fn=train_input_fn, steps=steps_per_period)
        predictions = linear_regressor.predict(input_fn=predict_input_fn)
        np_predictions = np.array([item['predictions'][0] for item in predictions])
        root_mean_squared_error = calc_rmse(np_predictions, targets)
        print("  period {:02d} : {:0.2f}".format(period, root_mean_squared_error))
        root_mean_squared_errors.append(root_mean_squared_error)
        plotter.add_period_line(linear_regressor, period)

    print('Model training finished')
    plotter.add_loss_metrics_plot(root_mean_squared_errors)
    plotter.show()
    callibration_data = gen_callibration_data(np_predictions, targets)
    display.display(callibration_data.describe())
    print('Final RMSE on training data: {:0.2f}'.format(root_mean_squared_error))
    return callibration_data


if __name__ == '__main__':
    train_model(
        learning_rate=0.1,
        steps=200,
        batch_size=1,
        input_feature='rooms_per_person'
    )

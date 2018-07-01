import math

# from IPython import display
# from matplotlib import cm
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


def get_training_dataset():
    dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
        sep=","
    )
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    # print(dataframe)
    # print(dataframe.describe())
    return dataframe


def get_test_dataset():
    dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_test.csv",
        sep=","
    )
    return dataframe


class ExampleTargets():

    def __init__(self, california_housing_dataframe):
        self.examples = self.preprocess_features(california_housing_dataframe)
        self.targets = self.preprocess_targets(california_housing_dataframe)
        self.target_series = self.targets['median_house_value']

    def preprocess_features(self, california_housing_dataframe):
        """
        Prepares input features from California housing data set.

        Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
        Returns:
        A DataFrame that contains the features to be used for the model, including
        synthetic features.
        """
        selected_features = california_housing_dataframe[[
            "latitude",
            "longitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income"
        ]]
        processed_features = selected_features.copy()
        # Create a synthetic feature.
        processed_features["rooms_per_person"] = (
            california_housing_dataframe["total_rooms"] /
            california_housing_dataframe["population"]
        )
        return processed_features

    def preprocess_targets(self, california_housing_dataframe):
        """
        Prepares target features (i.e., labels) from California housing data set.

        Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
        Returns:
        A DataFrame that contains the target feature.
        """
        output_targets = pd.DataFrame()
        # Scale the target to be in units of thousands of dollars.
        output_targets["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0
        )
        return output_targets

    def get_training_fn(self, batch_size):
        def training_fn():
            return my_input_fn(self, batch_size=batch_size)
        return training_fn

    def get_predict_fn(self):
        def predict_fn():
            return my_input_fn(self, num_epochs=1, shuffle=False)
        return predict_fn

    def get_prediction_rmse(self, linear_regressor):
        predictions = linear_regressor.predict(input_fn=self.get_predict_fn())
        np_predictions = np.array([item['predictions'][0] for item in predictions])
        return math.sqrt(metrics.mean_squared_error(np_predictions, self.targets))


def my_input_fn(features_targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    Trains a linear regression model of one feature

    Args:
        features_targets: ExampleTargets of features and targets
        batch_size: size of batches to be passed to the model
        shuffle: whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated.  None is
                    repeat indefinitely.
    Return:
        Tuple of (features, labels) for next data batch
    """
    # convert pandas data into dict of numpy arrays
    features = {key: np.array(value) for key, value in dict(features_targets.examples).items()}

    # construct a dataset and configure batch/repeating
    ds = Dataset.from_tensor_slices((features, features_targets.target_series))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def construct_feature_columns(input_features):
    """
    Construct the TensorFlow Feature Columns.

    Args:
        input_features: The names of the numerical input features to use.
    Returns:
        A set of feature columns
    """
    return set([tf.feature_column.numeric_column(f) for f in input_features])


def get_linear_regressor(learning_rate, training_examples):
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    return tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer,
    )


def train_model(
        learning_rate,
        steps,
        batch_size,
        training_data,
        validation_data,
):
    """
    Trains a linear regression model of multiple features.

    Args:
        learning_rate: `float`, the learning rate.
        steps: non-zero `int`, total number of training steps. A training step
               consists of a forward and backward pass using a single batch.
        batch_size: non-zero `int`, the batch size.
        training_data: An `ExampleTargets` containing a `DataFrame` containing
            one or more columns from `california_housing_dataframe` to use as
            input features for training, and a `DataFrame` containing exactly
            one column from `california_housing_dataframe` to use as target for
            training.
        validation_data: An `ExampleTargets` containing a `DataFrame` containing
            one or more columns from `california_housing_dataframe` to use as
            input features for validation, and a `DataFrame` containing exactly
            one column from `california_housing_dataframe` to use as target for
            validation.

    Returns:
        A `LinearRegressor` object trained on the training data.
    """
    periods = 10
    steps_per_period = steps // periods

    linear_regressor = get_linear_regressor(learning_rate, training_data.examples)
    train_input_fn = training_data.get_training_fn(batch_size)

    print('Training model')
    training_rmse_list = []
    validation_rmse_list = []
    for period in range(periods):
        linear_regressor.train(input_fn=train_input_fn, steps=steps_per_period)
        train_rmse = training_data.get_prediction_rmse(linear_regressor)
        validate_rmse = validation_data.get_prediction_rmse(linear_regressor)

        print("  period {:02d} : {:0.2f}".format(period, train_rmse))
        training_rmse_list.append(train_rmse)
        validation_rmse_list.append(validate_rmse)

    print('Model training finished')
    plot_rmse(training_rmse_list, validation_rmse_list)
    return linear_regressor


def evaluate_against_test_data(linear_regressor):
    california_housing_test_data = get_test_dataset()
    test_data = ExampleTargets(california_housing_test_data)
    test_rmse = test_data.get_prediction_rmse(linear_regressor)
    print('Test RMSE is {:0.2f}'.format(test_rmse))


def plot_single_lat_long(ax, examples, targets):
    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(
        examples["longitude"],
        examples["latitude"],
        cmap="coolwarm",
        c=targets["median_house_value"] / targets["median_house_value"].max()
    )


def plot_lat_long(validation_data, training_data):
    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")
    plot_single_lat_long(ax, validation_data.examples, validation_data.targets)

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Training Data")
    plot_single_lat_long(ax, training_data.examples, training_data.targets)

    plt.plot()
    plt.show()


def plot_rmse(training_rmse, validation_rmse):
    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()


def main():
    california_housing_dataframe = get_training_dataset()
    training_data = ExampleTargets(california_housing_dataframe.head(12000))
    validation_data = ExampleTargets(california_housing_dataframe.tail(5000))
    # plot_lat_long(validation_data, training_data)
    linear_regressor = train_model(
        # TWEAK THESE VALUES TO SEE HOW MUCH YOU CAN IMPROVE THE RMSE
        learning_rate=0.00003,
        steps=500,
        batch_size=5,
        training_data=training_data,
        validation_data=validation_data,
    )
    evaluate_against_test_data(linear_regressor)


if __name__ == '__main__':
    main()

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


def create_dataset(clip_rooms=False):
    dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
        sep=","
    )
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    # print(dataframe)
    # print(dataframe.describe())
    return dataframe


def preprocess_features(california_housing_dataframe):
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


def preprocess_targets(california_housing_dataframe):
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


def calc_rmse(np_predictions, targets):
    # convert to NumPy so we can calc error metrics
    return math.sqrt(metrics.mean_squared_error(np_predictions, targets))


def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets
):
    """
    Trains a linear regression model of multiple features.

    Args:
        learning_rate: `float`, the learning rate.
        steps: non-zero `int`, total number of training steps. A training step
               consists of a forward and backward pass using a single batch.
        batch_size: non-zero `int`, the batch size.
        training_examples: A `DataFrame` containing one or more columns from
            `california_housing_dataframe` to use as input features for training.
        training_targets: A `DataFrame` containing exactly one column from
            `california_housing_dataframe` to use as target for training.
        validation_examples: A `DataFrame` containing one or more columns from
            `california_housing_dataframe` to use as input features for validation.
        validation_targets: A `DataFrame` containing exactly one column from
            `california_housing_dataframe` to use as target for validation.

    Returns:
        A `LinearRegressor` object trained on the training data.
    """
    periods = 10
    steps_per_period = steps // periods

    linear_regressor = get_linear_regressor(learning_rate, training_examples)

    train_input_fn = lambda: my_input_fn(
        training_examples, training_targets['median_house_value'], batch_size=batch_size
    )
    predict_train_input_fn = lambda: my_input_fn(
        training_examples, training_targets['median_house_value'], num_epochs=1, shuffle=False
    )
    predict_validate_input_fn = lambda: my_input_fn(
        validation_examples, validation_targets['median_house_value'], num_epochs=1, shuffle=False
    )

    print('Training model')
    training_rmse_list = []
    validation_rmse_list = []
    for period in range(periods):
        linear_regressor.train(input_fn=train_input_fn, steps=steps_per_period)

        train_predictions = linear_regressor.predict(input_fn=predict_train_input_fn)
        np_train_predictions = np.array([item['predictions'][0] for item in train_predictions])
        train_rmse = calc_rmse(np_train_predictions, training_targets)

        validate_predictions = linear_regressor.predict(input_fn=predict_validate_input_fn)
        np_validate_predictions = np.array([item['predictions'][0] for item in validate_predictions])
        validate_rmse = calc_rmse(np_validate_predictions, validation_targets)

        print("  period {:02d} : {:0.2f}".format(period, train_rmse))
        training_rmse_list.append(train_rmse)
        validation_rmse_list.append(validate_rmse)

    print('Model training finished')
    plot_rmse(training_rmse_list, validation_rmse_list)
    return linear_regressor


def evaluate_against_test_data(linear_regressor):
    california_housing_test_data = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_test.csv",
        sep=","
    )
    test_examples = preprocess_features(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)
    predict_test_input_fn = lambda: my_input_fn(
        test_examples, test_targets['median_house_value'], num_epochs=1, shuffle=False
    )
    test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
    np_test_predictions = np.array([item['predictions'][0] for item in test_predictions])
    test_rmse = calc_rmse(np_test_predictions, test_targets)
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


def plot_lat_long(validation_examples, validation_targets, training_examples, training_targets):
    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")
    plot_single_lat_long(ax, validation_examples, validation_targets)

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Training Data")
    plot_single_lat_long(ax, training_examples, training_targets)

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
    california_housing_dataframe = create_dataset()
    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    # print(training_examples.describe())
    training_targets = preprocess_targets(california_housing_dataframe.head(12000))
    # print(training_targets.describe())
    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    # print(validation_examples.describe())
    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
    # print(validation_targets.describe())
    # plot_lat_long(validation_examples, validation_targets, training_examples, training_targets)
    linear_regressor = train_model(
        # TWEAK THESE VALUES TO SEE HOW MUCH YOU CAN IMPROVE THE RMSE
        learning_rate=0.00003,
        steps=500,
        batch_size=5,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets
    )
    evaluate_against_test_data(linear_regressor)


if __name__ == '__main__':
    main()

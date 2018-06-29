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


def create_dataset(clip_rooms=False):
    dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
        sep=","
    )
    # dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
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

    ax = plt.subplot(1,2,2)
    ax.set_title("Training Data")
    plot_single_lat_long(ax, training_examples, training_targets)

    plt.plot()
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
    plot_lat_long(validation_examples, validation_targets, training_examples, training_targets)


if __name__ == '__main__':
    main()

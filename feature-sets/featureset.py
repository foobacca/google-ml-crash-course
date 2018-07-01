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


def create_dataset():
    dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
        sep=","
    )
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
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


def correlation_report(training_data):
    correlation_dataframe = training_data.examples.copy()
    correlation_dataframe["target"] = training_data.target_series
    print(correlation_dataframe.corr())


def main():
    california_housing_dataframe = create_dataset()
    training_data = ExampleTargets(california_housing_dataframe.head(12000))
    validation_data = ExampleTargets(california_housing_dataframe.tail(5000))
    correlation_report(training_data)


if __name__ == '__main__':
    main()

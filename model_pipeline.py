import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import os
from typing import Tuple, Optional
from datetime import datetime

from utils import _setup_directory, _find_file


class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.X_train, self.X_test, self.y_train, self.y_test


def _preprocessing(df: pd.DataFrame) -> Dataset:

    # onehot encode categorical columns
    categorical_cols = ['Property Type']
    df = pd.get_dummies(df, columns=categorical_cols)

    # bucketise year-based columns
    # construction year
    if 'Year of construction' in df.columns:
        max_constr_yr = df['Year of construction'].max()
        construction_year_bins = [0, 1870, 1920, 1950, 1970, 1980, 1990, 2000, 2010, 2015, 2020, max_constr_yr] 
        construction_year_labels = np.arange(1,len(construction_year_bins))
        df['Year of construction'] = pd.cut(df['Year of construction'], bins=construction_year_bins, labels=construction_year_labels)
    # renovation year
    max_renov_yr = df['Renovation year'].max()
    renovation_year_bins = [0, 1980, 1990, 2000, 2010, 2015, 2020, max_renov_yr]
    renovation_year_labels = np.arange(1,len(renovation_year_bins))
    df['Renovation year'] = pd.cut(df['Renovation year'], bins=renovation_year_bins, labels=renovation_year_labels)

    # encode ordinal columns
    ordinal_columns = ['Energy class', 'Thermal insulation class']
    df[ordinal_columns] = OrdinalEncoder().fit_transform(df[ordinal_columns])

    # split into features and labels
    X = df.iloc[:, 1:].values
    y = np.array(df.iloc[:, 0].values)

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1)

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    data = Dataset(X_train_scaled, X_test_scaled, y_train, y_test)

    return data

def _create_model(num_features) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(input_shape=(num_features,), units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])

    return model


def bingobango(file: Optional[str] = None) -> Tuple[tf.keras.Sequential, tf.keras.callbacks.History]:

    # quick setup
    _setup_directory()

    # select most up to date set of clean data
    target_filepath, target_timestamp = _find_file('clean_datasets', file)

    # import clean data
    df = pd.read_csv(target_filepath)

    # preprocess data into a dataset
    X_train, X_test, y_train, y_test = _preprocessing(df).components()

    # create model with appropriate input layer size
    model = _create_model(X_train.shape[-1])

    # compile model
    optimizer = 'Adam'
    loss = 'mse'
    metrics = ['mae']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # train model
    epochs = 100
    BATCH_SIZE = 128
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=BATCH_SIZE)

    model_path = os.path.dirname(os.path.abspath(__file__)) + f'/models/model_{target_timestamp}'
    model.save(model_path)
    
    return model, hist


if __name__ == '__main__':
    bingobango()
    pass
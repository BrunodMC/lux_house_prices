import os
from typing import Optional, Tuple, Protocol
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_log_error

def _setup_directory() -> None:
    """Checks if required directories exist, creates them if not."""

    current_filepath = os.path.dirname(os.path.abspath(__file__))
    url_dir = current_filepath + '/extracted_URLs/'
    raw_csv_dir = current_filepath + '/raw_datasets/'
    clean_csv_dir = current_filepath + '/clean_datasets/'
    models_dir = current_filepath + '/models/'

    dirs = [url_dir, raw_csv_dir, clean_csv_dir, models_dir]

    # create directories if they do not exist
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
            print(f"Directory '{dir_}' created.")
    
    print('Ready.\n')

    return None

def _find_file(dirname: str, file: Optional[str] = None) -> Tuple[str, str]:
    
    current_filepath = os.path.dirname(os.path.abspath(__file__))
    dir_path = current_filepath + f'/{dirname}/'

    # if a filename is passed, try to find it
    if file:
        target_filepath = dir_path + file
        if not os.path.exists(target_filepath):
            raise Exception(f'Requested file ({target_filepath}) does not exist.')
        
        return target_filepath, ''

    # otherwise pick the most recent one
    list_of_files = os.listdir(dir_path)
    # check that it isn't empty
    if not len(list_of_files):
        raise Exception(f'No files exist in {dir_path}')
    
    file_timestamps = [int(x.split('_')[-1][:-4]) for x in list_of_files]
    target_timestamp = str(max(file_timestamps))
    target_filepath = dir_path + [file for file in list_of_files if target_timestamp in file][0]

    return target_filepath, target_timestamp

### Class purely used for type hinting for scikit models in the function 'evaluate_sk_model'
class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X) -> np.ndarray: ...
    def score(self, X, y, sample_weight=None): ...
    def get_params(self, **params): ...

# convenient evaluation function
def evaluate_sk_model(model: ScikitModel, 
                   X_valid: pd.DataFrame, 
                   y_valid: pd.Series | np.ndarray, 
                   X_train: Optional[pd.DataFrame] = None, 
                   y_train: Optional[pd.Series | np.ndarray] = None) -> dict[str, float]:
    """
    Runs model.predict() and prints a few evaluation metrics (MAE, RMSLE, R^2 score).
    """
    
    # dictionary to store results
    results = {}

    # generate training set predictions if provided
    if (X_train is not None) and (y_train is not None):
        train_preds = model.predict(X=X_train)
        print("Performance on Training Set:")
        train_mae = mean_absolute_error(y_true=y_train, y_pred=train_preds)
        results['Training MAE'] = train_mae
        print(f"\tMAE: {train_mae}")
        if len(train_preds[train_preds < 0]):
            print("\tNo RMSLE: predictions contain negative numbers")
        else:
            train_rmsle = root_mean_squared_log_error(y_true=y_train, y_pred=train_preds)
            results['Training RMSLE'] = train_rmsle
            print(f"\tRMSLE: {train_rmsle}")
        train_r2 = model.score(X=X_train, y=y_train)
        results['Training R^2'] = train_r2
        print(f"\tR^2: {train_r2}")
    
    # generate validation predictions
    valid_preds = model.predict(X=X_valid)
    print("Performance on Validation Set:")
    valid_mae = mean_absolute_error(y_true=y_valid, y_pred=valid_preds)
    results['Valid MAE'] = valid_mae
    print(f"\tMAE: {valid_mae}")
    if len(valid_preds[valid_preds < 0]):
        print("\tNo RMSLE: predictions containe negative numbers")
    else:
        valid_rmsle = root_mean_squared_log_error(y_true=y_valid, y_pred=valid_preds)
        results['Valid RMSLE'] = valid_rmsle
        print(f"\tRMSLE: {valid_rmsle}")
    valid_r2 = model.score(X=X_valid, y=y_valid)
    results['Valid R^2'] = valid_r2
    print(f"\tR^2: {valid_r2}")

    return results
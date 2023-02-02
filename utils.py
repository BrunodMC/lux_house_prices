import os
from typing import Optional, Tuple

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
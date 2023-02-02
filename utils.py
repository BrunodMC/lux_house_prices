import os

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
import os

def _setup_directory() -> None:
    """Checks if required directories exist, creates them if not."""

    current_filepath = os.path.dirname(os.path.abspath(__file__))
    url_dir = current_filepath + '/extracted_URLs/'
    raw_csv_dir = current_filepath + '/raw_datasets/'
    clean_csv_dir = current_filepath + '/clean_datasets/'

    # create directories if they do not exist
    if not os.path.exists(url_dir):
        os.makedirs(url_dir)
        print(f"Directory '{url_dir}' created.")
    
    if not os.path.exists(raw_csv_dir):
        os.makedirs(raw_csv_dir)
        print(f"Directory '{raw_csv_dir}' created.")

    if not os.path.exists(clean_csv_dir):
        os.makedirs(clean_csv_dir)
        print(f"Directory '{clean_csv_dir}' created.")
    
    print('Ready.\n')

    return None
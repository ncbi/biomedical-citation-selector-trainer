from . import config as cfg
from .helper import create_dir
import os.path
from urllib.request import urlretrieve


def run(workdir):
    START_DATA_FILE_NUM = 1
    END_DATA_FILE_NUM = cfg.NUM_BASLINE_FILES
    
    datadir = os.path.join(workdir, cfg.MEDLINE_DATA_DIR)
    create_dir(datadir)
    DOWNLOADED_DATA_FILEPATH_TEMPLATE = os.path.join(datadir, cfg.DOWNLOADED_DATA_FILENAME_TEMPLATE)
    URL_TEMPLATE = cfg.BASELINE_URL + cfg.BASELINE_FILENAME_TEMPLATE

    for file_num in range(START_DATA_FILE_NUM, END_DATA_FILE_NUM + 1):
        url = URL_TEMPLATE.format(file_num)
        filepath = DOWNLOADED_DATA_FILEPATH_TEMPLATE.format(file_num)
        print(f"{file_num}/{END_DATA_FILE_NUM}", end="\r")
        if os.path.isfile(filepath):
            continue
        urlretrieve(url, filepath)
        
    print(f"{file_num}/{END_DATA_FILE_NUM}")

    return cfg.NUM_BASLINE_FILES
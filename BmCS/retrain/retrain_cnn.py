from .cnn import train as train_cnn
from . import config as cfg
from .helper import load_dataset, load_pickled_object
import os.path


def run(workdir):

    DATA_DIR = workdir
    JOURNAL_ID_DICT_FILEPATH = os.path.join(workdir, cfg.JOURNAL_ID_DICT_FILENAME)
    RUNS_DIR = os.path.join(workdir, cfg.CNN_DATA_DIR)
    TRAIN_SET_FILEPATH = os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    VAL_SET_FILEPATH = os.path.join(workdir, cfg.VAL_SET_FILENAME)
    WORD_INDEX_DICT_FILEPATH = os.path.join(workdir, cfg.WORD_INDEX_DICT_FILENAME)
  
    word_index_lookup = load_pickled_object(WORD_INDEX_DICT_FILEPATH)
    journal_id_lookup = load_pickled_object(JOURNAL_ID_DICT_FILEPATH)
    train_set = load_dataset(TRAIN_SET_FILEPATH, cfg.ENCODING)
    val_set = load_dataset(VAL_SET_FILEPATH, cfg.ENCODING)

    train_cnn.run(DATA_DIR, RUNS_DIR, word_index_lookup, journal_id_lookup, train_set, val_set, cfg.PP_CONFIG)
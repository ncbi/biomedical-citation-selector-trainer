from .data_helper import DataGenerator, set_vocab_size
from .model import Model
from .settings import get_config


def run(data_dir, runs_dir, word_index_lookup, journal_id_lookup, test_set, preprocessing_config):
    
    config = get_config(data_dir=data_dir, runs_dir=runs_dir)

    input_dir = config.root_dir
    pp_config = config.inputs.preprocessing
    pred_config = config.pred_config

    pp_config.min_pub_year = preprocessing_config["min_pub_year"]
    pp_config.max_pub_year = preprocessing_config["max_pub_year"]
    pp_config.min_year_completed = preprocessing_config["min_year_indexed"]
    pp_config.max_year_completed = preprocessing_config["max_year_indexed"]
    pp_config.date_format = preprocessing_config["date_format"]

    word_index_lookup = set_vocab_size(word_index_lookup, pp_config.vocab_size)
    test_gen = DataGenerator(pp_config, word_index_lookup, journal_id_lookup, test_set, pred_config.batch_size, pred_config.limit)

    model = Model()
    model.restore(pred_config, input_dir)
    predictions = model.predict(pred_config, test_gen)
    return predictions
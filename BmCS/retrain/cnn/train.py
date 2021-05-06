from .data_helper import DataGenerator, set_vocab_size
from .model import Model
from .settings import get_config


def run(data_dir, runs_dir, word_index_lookup, journal_id_lookup, train_set, dev_set, preprocessing_config):
    
    config = get_config(data_dir=data_dir, runs_dir=runs_dir)

    output_dir = config.root_dir
    model_config = config.model
    ofs_config = config.train.optimize_fscore_threshold
    pp_config = config.inputs.preprocessing

    model_config.num_journals = len(journal_id_lookup) + 1
    pp_config.min_pub_year = preprocessing_config["min_pub_year"]
    pp_config.max_pub_year = preprocessing_config["max_pub_year"]
    
    pp_config.min_year_completed = preprocessing_config["min_year_indexed"]
    pp_config.max_year_completed = preprocessing_config["max_year_indexed"]
    pp_config.date_format = preprocessing_config["date_format"]

    word_index_lookup = set_vocab_size(word_index_lookup, pp_config.vocab_size)
    train_gen = DataGenerator(pp_config, word_index_lookup, journal_id_lookup, train_set, config.train.batch_size, config.train.train_limit)
    dev_gen =   DataGenerator(pp_config, word_index_lookup, journal_id_lookup, dev_set, config.train.batch_size, config.train.dev_limit)
    opt_gen =   DataGenerator(pp_config, word_index_lookup, journal_id_lookup, dev_set, ofs_config.batch_size, ofs_config.limit)

    model = Model()
    model.build(model_config)
    model.fit(config, train_gen, dev_gen, opt_gen, output_dir)
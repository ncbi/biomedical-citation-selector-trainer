from datetime import datetime as dt
from . import config as cfg
from .helper import load_delimited_data, load_indexing_periods
import json
import gzip
import os.path
import random


def create_dataset(workdir, num_xml_files):
    DATA_FILEPATH_TEMPLATE = os.path.join(workdir, cfg.MEDLINE_DATA_DIR, cfg.EXTRACTED_DATA_FILENAME_TEMPLATE)
    SELECTIVE_INDEXING_PERIODS_FILEPATH = os.path.join(workdir, cfg.SELECTIVE_INDEXING_PERIODS_FILENAME)

    data_set = {}
    indexing_periods = load_indexing_periods(SELECTIVE_INDEXING_PERIODS_FILEPATH, cfg.ENCODING, False)
    for file_num in range(1, num_xml_files + 1):
        print(f"{file_num}/{num_xml_files}", end="\r")
        data_filepath = DATA_FILEPATH_TEMPLATE.format(file_num)
        with gzip.open(data_filepath, "rt", encoding=cfg.ENCODING) as data_file: 
            data = json.load(data_file)
            for article in data["articles"]:
                if (is_selectively_indexed(indexing_periods, article) 
                    and not has_excluded_ref_type(article)
                    and was_completed_before_max_date(article)):
                    pmid = article["pmid"]
                    data_set[pmid] = article
    print(f"{num_xml_files}/{num_xml_files}")
    return data_set


def has_excluded_ref_type(article):
    for ref_type in article["ref_types"]:
        if ref_type in cfg.EXCLUDED_REF_TYPES:
            return True
    return False


def is_selectively_indexed(indexing_periods, article):
    nlm_id = article["journal_nlmid"]
    pub_year = article["pub_year"]
    if nlm_id not in indexing_periods:
        return False
    for indexing_period in indexing_periods[nlm_id]:
        if pub_year > indexing_period["start_year"] and (indexing_period["end_year"] is None or pub_year < indexing_period["end_year"]):
            return True
    return False


def is_test_set_candidate(reporting_nlmids, article):
    nlm_id = article["journal_nlmid"]
    pub_year = article["pub_year"]
    return pub_year == cfg.TEST_SET_YEAR and nlm_id in reporting_nlmids


def run(workdir, num_xml_files):
    TRAIN_SET_FILEPATH = os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    VAL_SET_FILEPATH = os.path.join(workdir, cfg.VAL_SET_FILENAME)
    TEST_SET_FILEPATH = os.path.join(workdir, cfg.TEST_SET_FILENAME)
    REPORTING_JOURNALS_FILEPATH = os.path.join(workdir, cfg.REPORTING_JOURNALS_FILENAME)

    data_set = create_dataset(workdir, num_xml_files)

    reporting_nlmids = [row[0] for row in load_delimited_data(REPORTING_JOURNALS_FILEPATH, cfg.ENCODING, ',')]
    test_set_candidates = [article for article in data_set.values() if is_test_set_candidate(reporting_nlmids, article)]
    other_articles =      [article for article in data_set.values() if not is_test_set_candidate(reporting_nlmids, article)]
    
    test_set_candidates = random.sample(test_set_candidates, len(test_set_candidates))
    test_set = test_set_candidates[:cfg.TEST_SET_SIZE]
    val_set = test_set_candidates[cfg.TEST_SET_SIZE:cfg.TEST_SET_SIZE + cfg.VAL_SET_SIZE]
    remaining_articles = test_set_candidates[cfg.TEST_SET_SIZE + cfg.VAL_SET_SIZE:]

    other_articles.extend(remaining_articles)
    train_set = random.sample(other_articles, len(other_articles))

    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")

    print("Saving train set...")
    save_dataset(train_set, TRAIN_SET_FILEPATH)
    print("Saving validation set...")
    save_dataset(val_set, VAL_SET_FILEPATH)
    print("Saving test set...")
    save_dataset(test_set, TEST_SET_FILEPATH)


def save_dataset(dataset, filepath):
    with gzip.open(filepath, "wt", encoding=cfg.ENCODING) as save_file:
         json.dump(dataset, save_file, ensure_ascii=False, indent=4)


def was_completed_before_max_date(article):
    date_completed = dt.strptime(article["date_completed"], cfg.DATE_FORMAT).date()
    result = date_completed <= cfg.MODEL_MAX_DATE
    return result
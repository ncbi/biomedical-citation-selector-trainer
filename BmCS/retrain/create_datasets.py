from . import config as cfg
import csv
from datetime import datetime as dt
from .helper import load_indexing_periods
import json
import gzip
import os.path
import random


def create_dataset(workdir, num_xml_files):
    DATA_FILEPATH_TEMPLATE = os.path.join(workdir, cfg.MEDLINE_DATA_DIR, cfg.EXTRACTED_DATA_FILENAME_TEMPLATE)
    SELECTIVE_INDEXING_PERIODS_FILEPATH = os.path.join(workdir, cfg.SELECTIVE_INDEXING_PERIODS_FILENAME)

    data_set = []
    indexing_periods = load_indexing_periods(SELECTIVE_INDEXING_PERIODS_FILEPATH, cfg.ENCODING, False)
    for file_num in range(1, num_xml_files + 1):
        print(f"{file_num}/{num_xml_files}", end="\r")
        data_filepath = DATA_FILEPATH_TEMPLATE.format(file_num)
        with gzip.open(data_filepath, "rt", encoding=cfg.ENCODING) as data_file: 
            data = json.load(data_file)
            for article in data["articles"]:
                if is_selectively_indexed(indexing_periods, article):
                    data_set.append(article)
    print(f"{num_xml_files}/{num_xml_files}")
    return data_set


def has_excluded_ref_type(article):
    for ref_type in article["ref_types"]:
        if ref_type in cfg.EXCLUDED_REF_TYPES:
            return True
    return False


def load_bmcs_pmids(path):
    pmids = set()
    with gzip.open(path, "rt", encoding=cfg.ENCODING) as file:
        for line in file:
            pmid, result = line.strip().split()
            pmid, result = int(pmid), int(result)
            if result < 20:
                pmids.add(pmid)
    return pmids


def load_problematic_journal_nlmids(path):
    with open(path, "rt", encoding=cfg.ENCODING) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        problematic_jouranls = set([str(row[0]) for row in csv_reader])
        return problematic_jouranls


def is_problematic_article(problematic_journal_nlmids, article):
    date_completed_str = article["date_completed"] if article["date_completed"] else article["date_revised"]
    year_completed = dt.strptime(date_completed_str, cfg.DATE_FORMAT).date().year
    
    nlm_id = article["journal_nlmid"]
    is_problematic_journal = nlm_id in problematic_journal_nlmids

    is_problematic = is_problematic_journal and year_completed < 2015 
    return is_problematic


def is_selectively_indexed(indexing_periods, article):
    nlm_id = article["journal_nlmid"]
    pub_year = article["pub_year"]
    if nlm_id not in indexing_periods:
        return False
    for indexing_period in indexing_periods[nlm_id]:
        if pub_year > indexing_period["start_year"] and (indexing_period["end_year"] is None or pub_year < indexing_period["end_year"]):
            return True
    return False


def run(workdir, num_xml_files):
    
    BMCS_RESULTS_FILEPATH = os.path.join(workdir, cfg.BMCS_RESULTS_FILENAME)
    PROBLEMATIC_JOURNALS_FILEPATH = os.path.join(workdir, cfg.PROBLEMATIC_JOURNALS_FILENAME)
    TRAIN_SET_FILEPATH = os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    VAL_SET_FILEPATH = os.path.join(workdir, cfg.VAL_SET_FILENAME)
    TEST_SET_FILEPATH = os.path.join(workdir, cfg.TEST_SET_FILENAME)

    bmcs_pmids = load_bmcs_pmids(BMCS_RESULTS_FILEPATH)

    data_set = create_dataset(workdir, num_xml_files)
    print(f"Dataset size: {len(data_set)}")

    data_set = [article for article in data_set if not has_excluded_ref_type(article)]
    print(f"Dataset size (exclude ref types): {len(data_set)}")

    problematic_journal_nlmids = load_problematic_journal_nlmids(PROBLEMATIC_JOURNALS_FILEPATH)
    data_set = [article for article in data_set if not is_problematic_article(problematic_journal_nlmids, article)]
    print(f"Dataset size (exclude problematic journals): {len(data_set)}")

    test_set_candidates = [article for article in data_set if article["pub_year"] == cfg.TEST_SET_YEAR]
    print(f"Test set candidate size: {len(test_set_candidates)}")

    test_set_candidates = [article for article in test_set_candidates if article["pmid"] in bmcs_pmids]
    print(f"Test set candidate size (BmCS pmids): {len(test_set_candidates)}")
    
    train_set_candidates = [article for article in data_set if article["pub_year"] < cfg.TEST_SET_YEAR]
    print(f"Train set candidate size: {len(train_set_candidates)}")

    train_set_candidates = [article for article in train_set_candidates if article["pmid"] not in bmcs_pmids]
    print(f"Train set candidate size (no BmCS pmids): {len(train_set_candidates)}")

    test_set_candidates = random.sample(test_set_candidates, len(test_set_candidates))
    test_set = test_set_candidates[:cfg.TEST_SET_SIZE]
    val_set = test_set_candidates[cfg.TEST_SET_SIZE:]
    print(f"Test set size: {len(test_set)}")
    print(f"Validation set size: {len(val_set)}")

    train_set = random.sample(train_set_candidates, len(train_set_candidates))
    print(f"Train set size: {len(train_set)}")

    print("Saving train set...")
    save_dataset(train_set, TRAIN_SET_FILEPATH)
    print("Saving validation set...")
    save_dataset(val_set, VAL_SET_FILEPATH)
    print("Saving test set...")
    save_dataset(test_set, TEST_SET_FILEPATH)


def save_dataset(dataset, filepath):
    with gzip.open(filepath, "wt", encoding=cfg.ENCODING) as save_file:
         json.dump(dataset, save_file, ensure_ascii=False, indent=4)
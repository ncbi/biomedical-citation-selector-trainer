from . import config as cfg
import csv
from datetime import datetime as dt
from .helper import load_dataset, load_indexing_periods, parse_date
import json
import gzip
import os.path
import random


def add_bmcs_processing_data(data_set, bmcs_data_lookup):
    for article in data_set:
        pmid = article["pmid"]
        if pmid in bmcs_data_lookup:
            article_bmcs_data = bmcs_data_lookup[pmid]
            article["bmcs_processed_date"] = article_bmcs_data["bmcs_processed_date"]
            article["bmcs_result"] = article_bmcs_data["bmcs_result"]
        else:
            article["bmcs_processed_date"] = None
            article["bmcs_result"] =  None


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
                if is_selectively_indexed(indexing_periods, article):
                    pmid = article["pmid"]
                    data_set[pmid] = article
    print(f"{num_xml_files}/{num_xml_files}")
    data_set_list = list(data_set.values())
    return data_set_list


def has_excluded_ref_type(article):
    for ref_type in article["ref_types"]:
        if ref_type in cfg.EXCLUDED_REF_TYPES:
            return True
    return False


def load_bmcs_processing_data(path):
    lookup = {}
    with gzip.open(path, "rt", encoding=cfg.ENCODING) as file:
        for line in file:
            line_data = line.strip().split(sep="\t")
            pmid = int(line_data[0])
            processed_date_str = parse_date(line_data[3], cfg.BMCS_RESULTS_DATE_FORMAT).isoformat()
            bmcs_result = int(line_data[4])
            lookup[pmid] =  { "bmcs_processed_date": processed_date_str, "bmcs_result": bmcs_result }
    return lookup


def load_problematic_journal_nlmids(path):
    with open(path, "rt", encoding=cfg.ENCODING) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        problematic_jouranls = set([str(row[0]) for row in csv_reader])
        return problematic_jouranls


def is_problematic_article(problematic_journal_nlmids, article):
    if not article["date_completed"]:
        return False

    date_completed_str = article["date_completed"]
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


def is_bmcs_manual_labeled(article):
    result = ((article["bmcs_result"] is not None) and
              (article["bmcs_result"] == cfg.BMCS_UNCERTAIN_RESULT) and
              (parse_date(article["bmcs_processed_date"], cfg.DATE_FORMAT) <= cfg.MAX_PROCESSED_DATE))
    return result


def is_manual_labeled(article):
    result = ((article["bmcs_result"] is None) and
              (article["date_completed"] is not None) and
              (parse_date(article["date_completed"], cfg.DATE_FORMAT) <= cfg.MAX_PROCESSED_DATE))
    return result


def run(workdir, num_xml_files):
    
    BMCS_RESULTS_FILEPATH = os.path.join(workdir, cfg.BMCS_RESULTS_FILENAME)
    PROBLEMATIC_JOURNALS_FILEPATH = os.path.join(workdir, cfg.PROBLEMATIC_JOURNALS_FILENAME)
    TRAIN_SET_FILEPATH = os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    VAL_SET_FILEPATH = os.path.join(workdir, cfg.VAL_SET_FILENAME)
    SELECTIVELY_INDEXED_JOURNALS_FILEPATH = os.path.join(workdir, cfg.SELECTIVELY_INDEXED_JOURNALS_FILENAME)
    TEST_SET_FILEPATH = os.path.join(workdir, cfg.TEST_SET_FILENAME)

    data_set = create_dataset(workdir, num_xml_files)
    print(f"Dataset size: {len(data_set)}")

    bmcs_processing_data = load_bmcs_processing_data(BMCS_RESULTS_FILEPATH)
    add_bmcs_processing_data(data_set, bmcs_processing_data)

    data_set = [article for article in data_set if article["pub_year"] <= cfg.TEST_SET_YEAR]
    print(f"Dataset size (exclude published after test year): {len(data_set)}")

    data_set = [article for article in data_set if not has_excluded_ref_type(article)]
    print(f"Dataset size (exclude ref types): {len(data_set)}")

    problematic_journal_nlmids = load_problematic_journal_nlmids(PROBLEMATIC_JOURNALS_FILEPATH)
    data_set = [article for article in data_set if not is_problematic_article(problematic_journal_nlmids, article)]
    print(f"Dataset size (exclude problematic journals): {len(data_set)}")

    if not cfg.USE_EXISTING_VAL_TEST_SETS:
        test_set_candidates = [article for article in data_set if article["pub_year"] == cfg.TEST_SET_YEAR]
        print(f"Test set candidate size: {len(test_set_candidates)}")

        selectively_indexed_journals = json.load(open(SELECTIVELY_INDEXED_JOURNALS_FILEPATH, "rt", encoding=cfg.ENCODING))
        selectively_indexed_journal_nlmids = set(selectively_indexed_journals.keys())

        test_set_candidates = [article for article in test_set_candidates if article["journal_nlmid"] in selectively_indexed_journal_nlmids]
        print(f"Test set candidate size (selectively indexed journals): {len(test_set_candidates)}")
    
        test_set_candidates = [article for article in test_set_candidates if article["bmcs_processed_date"]]
        print(f"Test set candidate size (bmcs processed date): {len(test_set_candidates)}")

        test_set_candidates = [article for article in test_set_candidates if parse_date(article["bmcs_processed_date"], cfg.DATE_FORMAT) <= cfg.MAX_PROCESSED_DATE ]
        print(f"Test set candidate size (exclude after max bmcs processed date): {len(test_set_candidates)}")

        test_set_candidates = random.sample(test_set_candidates, len(test_set_candidates))
        val_test_set_size = cfg.VAL_SET_SIZE + cfg.TEST_SET_SIZE
        test_set = test_set_candidates[:cfg.TEST_SET_SIZE]
        val_set = test_set_candidates[cfg.TEST_SET_SIZE:val_test_set_size]

        print("Saving validation set...")
        save_dataset(val_set, VAL_SET_FILEPATH)
        print("Saving test set...")
        save_dataset(test_set, TEST_SET_FILEPATH)

    test_set = load_dataset(TEST_SET_FILEPATH, cfg.ENCODING)
    val_set = load_dataset(VAL_SET_FILEPATH, cfg.ENCODING)
    print(f"Test set size: {len(test_set)}")
    print(f"Validation set size: {len(val_set)}")

    val_test_set_pmids =  [a["pmid"] for a in val_set] 
    val_test_set_pmids += [a["pmid"] for a in test_set]
    val_test_set_pmids = set(val_test_set_pmids)
    print(f"Val test set pmid count: {len(val_test_set_pmids)}")

    train_set_candidates = [article for article in data_set if is_manual_labeled(article) or is_bmcs_manual_labeled(article) ]
    print(f"Train set candidate size: {len(train_set_candidates)}")

    train_set_candidates = [article for article in train_set_candidates if article["pmid"] not in val_test_set_pmids]
    print(f"Train set candidate size (no val test set pmids): {len(train_set_candidates)}")

    train_set = random.sample(train_set_candidates, len(train_set_candidates))
    print(f"Train set size: {len(train_set)}")

    print("Saving train set...")
    save_dataset(train_set, TRAIN_SET_FILEPATH)


def save_dataset(dataset, filepath):
    with gzip.open(filepath, "wt", encoding=cfg.ENCODING) as save_file:
         json.dump(dataset, save_file, ensure_ascii=False, indent=4)
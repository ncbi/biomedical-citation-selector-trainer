from datetime import date

# User settings
BASELINE_FILENAME_TEMPLATE = "pubmed23n{0:04d}.xml.gz"
BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/" # Must include last /
BMCS_RESULTS_FILENAME = "selective-type-dump_15th_Nov_22.txt.gz"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_DELAY = 0.34 # seconds
EUTILS_RETMAX = 10000
JOURNAL_MEDLINE_FILENAME = "J_Medline_7th_Nov_22.txt.gz"
MAX_PROCESSED_DATE = date(2021, 12, 7) # Susan Schmidt: for the majority of the selective journals, the effective date when they were no longer manually selected was 12.08.2021
NUM_BASLINE_FILES = 1166
PROBLEMATIC_JOURNALS_FILENAME = "problematic_journals.csv"
SELECTIVE_INDEXING_PERIODS_FILENAME = "2022_selective_indexing_periods_input.csv"
SELECTIVELY_INDEXED_JOURNALS_FILENAME = "selectively_indexed_journals_21st_Dec_22.json"
TEST_SET_FILENAME = "test_set.json.gz"
TEST_SET_SIZE = 15000
TEST_SET_YEAR = 2021
USE_EUTILS = False # Potential issue that eutils can only download 10k records at a time. Also, no error handing has been implemented for eutils services.
USE_EXISTING_VAL_TEST_SETS = False
VAL_SET_FILENAME = "validation_set.json.gz"
VAL_SET_SIZE = 15000


# System settings
BMCS_RESULTS_DATE_FORMAT = "%Y/%m/%d"
BMCS_UNCERTAIN_RESULT = 2
CNN_DATA_DIR = "cnn_model"
DATE_FORMAT = "%Y-%m-%d"
DOWNLOADED_DATA_FILENAME_TEMPLATE = "pubmed23n{0:04d}.xml.gz"
ENCODING = "utf8"
EXTRACTED_DATA_FILENAME_TEMPLATE = "{0:04d}.json.gz"
EXCLUDED_REF_TYPES = ["CommentOn",
"ErratumFor",
"ExpressionOfConcernFor",
"RepublishedFrom",
"RetractionOf",
"UpdateOf",
"OriginalReportIn"
"ReprintOf"]
JOURNAL_ID_DICT_FILENAME = "journal_id_dict.pkl"
JOURNAL_ID_TXT_FILENAME = "journal_ids.txt"
MEDLINE_DATA_DIR = "medline_data"
OPT_THRESHOLDS_DELTA = .00005
OPT_THRESHOLDS_FILENAME_TEMPLATE = "{}_optimum_thresholds.txt"
OPT_THRESHOLDS_TARGET_PRECISION = 0.988 # 0.97 # These targets have been updated to match the measured precision/recall of BmCS v1 on an updated 2018 test set with a real-world distribution of articles.
OPT_THRESHOLDS_TARGET_RECALL = 0.983 # 0.995
TRAIN_SET_FILENAME = "train_set.json.gz"
VOTING_DATA_DIR = "voting_model"
VOTING_MODEL_FILENAME = "voting_model.joblib"
WORD_INDEX_DICT_FILENAME = "word_index_dict.pkl"
WORD_INDEX_TXT_FILENAME = "word_indices.txt"
WORD_INDEX_TXT_VOCAB_SIZE = 400000

VOTING_TRAIN_YEARS = [2021, 2020, 2019, 2018, 2017]

PP_CONFIG = { "min_pub_year": 1809, 
              "max_pub_year": TEST_SET_YEAR, 
              "min_year_indexed": 1965, 
              "max_year_indexed": TEST_SET_YEAR, 
              "date_format": DATE_FORMAT }
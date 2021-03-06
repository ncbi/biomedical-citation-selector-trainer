from datetime import date

# User settings
BASELINE_FILENAME_TEMPLATE = "pubmed18n{0:04d}.xml.gz"
BASELINE_URL = "https://mbr.nlm.nih.gov/Download/Baselines/2018/" # Must include last /
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_DELAY = 0.34 # seconds
EUTILS_RETMAX = 10000
JOURNAL_MEDLINE_FILENAME = "J_Medline.txt"
MODEL_MAX_DATE = date(2018,8,31)
NUM_BASLINE_FILES = 1250
REPORTING_JOURNALS_FILENAME = "selectively-indexed-journals-of-interest.csv"
SERIALS_FILENAME = "lsi2018.xml"
USE_EUTILS = True
VOTING_TRAIN_YEAR = 2017

# System settings
CNN_DATA_DIR = "cnn_model"
DATE_FORMAT = "%Y-%m-%d"
DOWNLOADED_DATA_FILENAME_TEMPLATE = "{0:04d}.xml.gz"
ENCODING = "utf8"
EXTRACTED_DATA_FILENAME_TEMPLATE = "{0:04d}.json.gz"
EXCLUDED_REF_TYPES = ['CommentOn',
'ErratumFor',
'ExpressionOfConcernFor',
'RepublishedFrom',
'RetractionOf',
'UpdateOf',
'OriginalReportIn'
'ReprintOf']
JOURNAL_ID_DICT_FILENAME = "journal_id_dict.pkl"
JOURNAL_ID_TXT_FILENAME = "journal_ids.txt"
MEDLINE_DATA_DIR = "medline_data"
MIN_PUB_YEAR = 1809
MIN_YEAR_INDEXED = 1965
OPT_THRESHOLDS_DELTA = .00005
OPT_THRESHOLDS_FILENAME_TEMPLATE = "{}_optimum_thresholds.txt"
OPT_THRESHOLDS_TARGET_PRECISION = 0.97
OPT_THRESHOLDS_TARGET_RECALL = 0.995
SELECTIVE_INDEXING_PERIODS_FILENAME = "selective_indexing_periods.txt"
TEST_SET_FILENAME = "test_set.json.gz"
TEST_SET_SIZE = 30000
TRAIN_SET_FILENAME = "train_set.json.gz"
VAL_SET_FILENAME = "val_set.json.gz"
VAL_SET_SIZE = 15000
VOTING_DATA_DIR = "voting_model"
VOTING_MODEL_FILENAME = "voting_model.joblib"
WORD_INDEX_DICT_FILENAME = "word_index_dict.pkl"
WORD_INDEX_TXT_FILENAME = "word_indices.txt"
WORD_INDEX_TXT_VOCAB_SIZE = 400000

# Dynamic settings
PP_CONFIG = {"min_pub_year": MIN_PUB_YEAR,
             "min_year_indexed": MIN_YEAR_INDEXED,
             "model_max_year": MODEL_MAX_DATE.year,
             "date_format": DATE_FORMAT}
TEST_SET_YEAR = MODEL_MAX_DATE.year

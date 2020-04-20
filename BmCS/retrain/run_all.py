from . import create_datasets
from . import create_journal_id_lookups
from . import create_word_index_lookups
from . import determine_optimum_thresholds
from . import download_medline_baseline
from . import extract_journal_indexing_periods
from . import extract_medline_data
from . import retrain_cnn
from . import retrain_voting


def run(workdir):
    print("Extracting Journal Selective Indexing Periods...")
    extract_journal_indexing_periods.run(workdir)
    print("Downloading MEDLINE baseline...")
    download_medline_baseline.run(workdir)
    print("Extracting MEDLINE data...")
    extract_medline_data.run(workdir)
    print("Creating datasets...")
    create_datasets.run(workdir)
    print("Create word index lookups...")
    create_word_index_lookups.run(workdir)
    print("Create journal id lookups...")
    create_journal_id_lookups.run(workdir)
    print("Retraining voting model...")
    retrain_voting.run(workdir)
    print("Retraining CNN model...")
    retrain_cnn.run(workdir)
    print("Finding optimum thresholds...")
    determine_optimum_thresholds.run(workdir)
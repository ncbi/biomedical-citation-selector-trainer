from .cnn import pred as cnn_pred
from . import config as cfg
from .helper import load_dataset, load_pickled_object, preprocess_voting_model_data
import joblib
import numpy as np
import os.path
from sklearn.metrics import precision_score, recall_score


def compute_precision_recall(predictions, threshold):
    y_true = predictions[:,0]
    y_pred = predictions[:,1]
    y_pred_thresh = y_pred >= threshold
    y_pred_thresh = y_pred_thresh.astype('float32')

    if (np.sum(y_pred_thresh) == 0.):
        return 1., 0.

    precision = precision_score(y_true, y_pred_thresh, average='binary')
    recall = recall_score(y_true, y_pred_thresh, average='binary')
    return precision, recall


def get_combined_predictions(cnn_predictions, voting_predictions):
    combined_predictions = {}
    for pmid in cnn_predictions:
        cnn_prediction = cnn_predictions[pmid]
        voting_prediction = voting_predictions[pmid]
        act = voting_prediction['act']
        cnn_score = cnn_prediction['score']
        voting_score = voting_prediction['score']
        combined_score = voting_score*cnn_score
        combined_predictions[pmid] = { 'act': act, 'score': combined_score}
    return combined_predictions


def get_cnn_predictions(workdir, val_set):
    CNN_DATA_DIR = workdir
    CNN_RUNS_DIR = os.path.join(workdir, cfg.CNN_DATA_DIR)
    JOURNAL_ID_DICT_FILEPATH = os.path.join(workdir, cfg.JOURNAL_ID_DICT_FILENAME)
    WORD_INDEX_DICT_FILEPATH = os.path.join(workdir, cfg.WORD_INDEX_DICT_FILENAME)

    word_index_lookup = load_pickled_object(WORD_INDEX_DICT_FILEPATH)
    journal_id_lookup = load_pickled_object(JOURNAL_ID_DICT_FILEPATH)
    predictions = cnn_pred.run(CNN_DATA_DIR, CNN_RUNS_DIR, word_index_lookup, journal_id_lookup, val_set, cfg.PP_CONFIG)
    return predictions


def get_voting_predictions(workdir, val_set):
    VOTING_MODEL_FILEPATH = os.path.join(workdir, cfg.VOTING_DATA_DIR, cfg.VOTING_MODEL_FILENAME)

    model = joblib.load(VOTING_MODEL_FILEPATH)
    val_data = preprocess_voting_model_data(val_set)
    scores = model.predict_proba(val_data)

    predictions = {}
    for idx in range(len(val_set)):
        article = val_set[idx]
        pmid = article['pmid']
        act = float(article['is_indexed'])
        score = scores[idx, 0]
        predictions[pmid] = { 'act': act, 'score': score }
    return predictions


# def _save_test_set_predictions(workdir, filename,  combined_predictions):
#     import gzip
    
#     BMCS_RESULTS_FILEPATH = os.path.join(workdir, cfg.BMCS_RESULTS_FILENAME)
#     SAVE_FILEPATH = os.path.join(workdir, filename)
    
#     with gzip.open(BMCS_RESULTS_FILEPATH, "rt", encoding=cfg.ENCODING) as read_file, \
#          open(SAVE_FILEPATH, "wt", encoding=cfg.ENCODING) as write_file:
#         write_file.write("pmid,is_indexed,bmcs_result,combined_pred\n")
#         for line in read_file:
#             pmid, result = line.strip().split()
#             pmid, result = int(pmid), int(result)
#             if pmid in combined_predictions:
#                 write_file.write(f"{pmid},{int(combined_predictions[pmid]['act'])},{result},{float(combined_predictions[pmid]['score']):.10f}\n")
            

def run(workdir):
    OPT_THRESHOLDS_FILEPATH_TEMPLATE = os.path.join(workdir, cfg.OPT_THRESHOLDS_FILENAME_TEMPLATE)
    VAL_SET_FILEPATH = os.path.join(workdir, cfg.VAL_SET_FILENAME)

    val_set = load_dataset(VAL_SET_FILEPATH, cfg.ENCODING)
    #val_set = [c for c in val_set if c["journal_nlmid"] != "101653440" ] # v3 exclude Sci Adv due to false negatives
    cnn_predictions = get_cnn_predictions(workdir, val_set)
    voting_predictions = get_voting_predictions(workdir, val_set)
    combined_predictions = get_combined_predictions(cnn_predictions, voting_predictions)

    #_save_test_set_predictions(workdir, "val_set_cnn_predictions.csv", cnn_predictions)
    #_save_test_set_predictions(workdir, "val_set_voting_predictions.csv", voting_predictions)
    #_save_test_set_predictions(workdir, "val_set_predictions.csv", combined_predictions)
    
    cnn_predictions = to_numpy(cnn_predictions)
    voting_predictions = to_numpy(voting_predictions)
    combined_predictions = to_numpy(combined_predictions)

    #np.save("cnn_predictions.npy", cnn_predictions)
    #np.save("voting_predictions.npy", voting_predictions)
    #np.save("combined_predictions.npy", combined_predictions)

    filepath = OPT_THRESHOLDS_FILEPATH_TEMPLATE.format("cnn")
    save_optimum_thresholds(cnn_predictions, filepath)

    filepath = OPT_THRESHOLDS_FILEPATH_TEMPLATE.format("voting")
    save_optimum_thresholds(voting_predictions, filepath)

    filepath = OPT_THRESHOLDS_FILEPATH_TEMPLATE.format("combined")
    save_optimum_thresholds(combined_predictions, filepath)


def save_optimum_thresholds(predictions, save_filepath):
    
    precision, recall, threshold = 0., 1., 0.
    while (recall > cfg.OPT_THRESHOLDS_TARGET_RECALL):
        last_precision, last_recall, last_threshold = precision, recall, threshold
        threshold += cfg.OPT_THRESHOLDS_DELTA
        precision, recall = compute_precision_recall(predictions, threshold)
    
    with open(save_filepath, 'wt', encoding=cfg.ENCODING) as file:
        file.write(f"Threshold: {last_threshold}, Precision: {last_precision}, Recall: {last_recall}\n")

    precision, recall, threshold = 1., 0., 1.
    while (precision > cfg.OPT_THRESHOLDS_TARGET_PRECISION):
        last_precision, last_recall, last_threshold = precision, recall, threshold
        threshold -= cfg.OPT_THRESHOLDS_DELTA
        precision, recall = compute_precision_recall(predictions, threshold)
    
    with open(save_filepath, 'at', encoding=cfg.ENCODING) as file:
        file.write(f"Threshold: {last_threshold}, Precision: {last_precision}, Recall: {last_recall}\n")


def to_numpy(predictions):
    array = np.array([[prediction['act'], prediction['score']] for prediction in predictions.values()])
    return array
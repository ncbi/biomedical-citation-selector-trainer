from . import config as cfg
from .helper import create_dir, load_dataset, preprocess_voting_model_data
from ..item_select import ItemSelector
import os.path
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline, FeatureUnion


def get_pipeline(model):
    pipeline = Pipeline([
        ("union", FeatureUnion(
            transformer_list=[
                ("titles_pipe", Pipeline([
                    ("selector", ItemSelector(column="titles")),
                    ("tfidf", TfidfVectorizer()),
                ])),
                ("author_pipe", Pipeline([
                    ("selector", ItemSelector(column="author_list")),
                    ("tfidf", TfidfVectorizer()),
                ])),
                ("abstract_pipe", Pipeline([
                    ("selector", ItemSelector(column="abstract")),
                    ("tfidf", TfidfVectorizer()),
                ])),
                ],
            )),
            ("ensemble", model),
        ])
    return pipeline


def run(workdir):
    TRAIN_SET_FILEPATH = os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    save_dir = os.path.join(workdir, cfg.VOTING_DATA_DIR)
    create_dir(save_dir)
    SAVE_FILEPATH = os.path.join(save_dir, cfg.VOTING_MODEL_FILENAME)
    
    train_set = load_dataset(TRAIN_SET_FILEPATH, cfg.ENCODING)
    training_data = preprocess_voting_model_data(train_set, cfg.VOTING_TRAIN_YEAR)

    models = [
        ("sgd", SGDClassifier(loss="modified_huber", alpha=.0001, max_iter=1000)),
        ("lg", LogisticRegression(C=2, random_state=0)),
        ("bnb", BernoulliNB(alpha=.01)),
        ("rfc", RandomForestClassifier(n_estimators=100, criterion="gini", random_state=0))
        ]

    voting_model = VotingClassifier(estimators=models, voting="soft", n_jobs=8)
    pipeline = get_pipeline(voting_model)
    pipeline.fit(training_data, training_data["labels"])
    joblib.dump(pipeline, SAVE_FILEPATH)
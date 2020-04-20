from collections import defaultdict
from . import config as cfg
import gzip
import json
import os.path
from nltk.tokenize import word_tokenize
from pickle import dump


OFFSET = 2


def create_dict(train_set):       
    token_counts = defaultdict(int)
    num_articles = len(train_set)
    for idx, article in enumerate(train_set):
        if idx % 10000 == 0: 
            print(f"{idx}/{num_articles}", end="\r")
        title = article["title"]
        abstract = article["abstract"]
        title_tokens = tokenize(title)
        abstract_tokens = tokenize(abstract)
        tokens = title_tokens + abstract_tokens
        for token in tokens:
            token_counts[token] += 1
    print(f"{num_articles}/{num_articles}")
    sorted_items = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    lookup = { item[0]: index + OFFSET for index, item in enumerate(sorted_items) }
    return lookup


def run(workdir):
    TRAIN_SET_FILEPATH = os.path.join(workdir, cfg.TRAIN_SET_FILENAME)
    WORD_INDEX_DICT_FILEPATH = os.path.join(workdir, cfg.WORD_INDEX_DICT_FILENAME)
    WORD_INDEX_TXT_FILEPATH = os.path.join(workdir, cfg.WORD_INDEX_TXT_FILENAME)

    with gzip.open(TRAIN_SET_FILEPATH , "rt", encoding=cfg.ENCODING) as train_file:
        train_set = json.load(train_file)
        word_index_dict = create_dict(train_set)

    with open(WORD_INDEX_DICT_FILEPATH, "wb") as wid_file:
        dump(word_index_dict, wid_file)

    vocab_size = cfg.WORD_INDEX_TXT_VOCAB_SIZE - 2 # Minus unknown and padding
    with open(WORD_INDEX_TXT_FILEPATH, "wt") as wit_file:
        for word, index in sorted(word_index_dict.items(), key=lambda x: x[1])[:vocab_size]:
            wit_file.write(f"{index}\t{word}\n")


def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens
import gzip
import json
import os
import pickle


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def load_dataset(path, encoding):
    with gzip.open(path , "rt", encoding=encoding) as file:
        dataset = json.load(file)
        return dataset


def load_delimited_data(path, encoding, delimiter):
    with open(path, "rt", encoding=encoding) as file:
        data = tuple( tuple(data_item.strip() for data_item in line.strip().split(delimiter)) for line in file ) 
    return data


def load_pickled_object(path):
    loaded_object = pickle.load(open(path, "rb"))
    return loaded_object


def preprocess_voting_model_data(data, year_filter=None):
    training_data = { "titles" : [], "abstract" : [], "author_list" : [], "labels": [] }
    for article in data:
        if year_filter and article['pub_year'] != year_filter:
            continue
        training_data["titles"].append(article["title"])
        training_data["abstract"].append(article["abstract"])
        training_data["author_list"].append(article["affiliations"])
        training_data["labels"].append(not article["is_indexed"])
    return training_data


def save_delimited_data(path, encoding, delimiter, data):
    with open(path, "wt", encoding=encoding) as file:
        for data_row in data:
            line = delimiter.join([str(data_item) for data_item in data_row]) + "\n"
            file.write(line)
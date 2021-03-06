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


def load_indexing_periods(filepath, encoding, is_fully_indexed):
    periods = {}
    with open(filepath, "rt", encoding=encoding) as file:
        for line in file:
            split = line.split(",")

            nlm_id = split[0].strip()
            citation_subset = split[1].strip()
            start_year = int(split[2].strip())
            end_year = int(split[3].strip())
            
            if start_year < 0:
                continue
            if end_year < 0:
                end_year = None

            period = { "citation_subset": citation_subset, "is_fully_indexed": is_fully_indexed, "start_year": start_year, "end_year": end_year }
            if nlm_id in periods:
                periods[nlm_id].append(period)
            else:
                periods[nlm_id] = [period]
    return periods


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
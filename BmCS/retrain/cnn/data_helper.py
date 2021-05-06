from datetime import datetime as dt
import math
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


def set_vocab_size(word_index_lookup, vocab_size):
    sorted_items = sorted(word_index_lookup.items(), key=lambda x: x[1])
    word_index_lookup_size = vocab_size - 2 # minus unknown/padding
    modified_lookup = dict(sorted_items[:word_index_lookup_size])
    assert(len(modified_lookup) == word_index_lookup_size)
    return modified_lookup


def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens


class DataGenerator(Sequence):

    def __init__(self, pp_config, word_index_lookup, journal_id_lookup, data_set, batch_size, max_examples = 1000000000, tokenizer=tokenize):
        self._pp_config = pp_config
        self._word_index_lookup = word_index_lookup
        self._journal_id_lookup = journal_id_lookup
        self._data_set = data_set
        self._batch_size = batch_size 
        self._tokenizer = tokenizer
        self._num_examples = min(len(data_set), max_examples)
   
    def __len__(self):
        length = int(math.ceil(self._num_examples/self._batch_size))
        return length

    def __getitem__(self, idx):
        batch_start_index = idx * self._batch_size
        batch_end_index = (idx + 1) * self._batch_size

        pmid, title, abstract, pub_year, year_completed, journal_id, is_indexed = self._get_batch_data(batch_start_index, batch_end_index)
        
        title_input = self._vectorize_batch_text(title, self._pp_config.title_max_words)
        abstract_input = self._vectorize_batch_text(abstract, self._pp_config.abstract_max_words)

        pub_year = np.array(pub_year, dtype=np.int32).reshape(-1, 1)
        pub_year_indices = pub_year - self._pp_config.min_pub_year
        pub_year_input = self._to_time_period_input(pub_year_indices, self._pp_config.num_pub_year_time_periods)

        year_completed = np.array(year_completed, dtype=np.int32).reshape(-1, 1)
        year_completed_indices = year_completed - self._pp_config.min_year_completed
        year_completed_input = self._to_time_period_input(year_completed_indices, self._pp_config.num_year_completed_time_periods)

        journal_input = np.array(journal_id, dtype=np.int32).reshape(-1, 1)

        pmid_input = np.array(pmid, dtype=np.int32).reshape(-1, 1)

        batch_x = { 'pmids': pmid_input, 'title_input': title_input, 'abstract_input': abstract_input, 'pub_year_input': pub_year_input, 'year_completed_input': year_completed_input, 'journal_input': journal_input}
    
        is_indexed = np.array(is_indexed, dtype=np.float32).reshape(-1, 1)

        batch_y = is_indexed
        
        return batch_x, batch_y

    def _get_batch_data(self, start_index, end_index):
        inputs = []
        for article in self._data_set[start_index: end_index]:
            pmid = article["pmid"]
            title = article["title"]
            abstract = article["abstract"]
            
            pub_year = article["pub_year"]
            pub_year = self._pp_config.max_pub_year if pub_year > self._pp_config.max_pub_year else pub_year
            pub_year = self._pp_config.min_pub_year if pub_year < self._pp_config.min_pub_year else pub_year
            
            date_completed_str = article['date_completed'] if article['date_completed'] else article['date_revised']
            year_completed = dt.strptime(date_completed_str, self._pp_config.date_format).date().year
            year_completed = self._pp_config.max_year_completed if year_completed > self._pp_config.max_year_completed else year_completed
            year_completed = self._pp_config.min_year_completed if year_completed < self._pp_config.min_year_completed else year_completed
            
            nlmid = article["journal_nlmid"]
            journal_id = self._journal_id_lookup[nlmid] if nlmid in self._journal_id_lookup else self._pp_config.unknown_journal_index
            is_indexed = article["is_indexed"]
            
            inputs.append([pmid, title, abstract, pub_year, year_completed, journal_id, is_indexed])
        return zip(*inputs)

    def _to_time_period_input(self, year_indices, num_time_periods):
        batch_size = year_indices.shape[0]
        batch_indices = np.zeros([batch_size, num_time_periods], np.int32)
        batch_indices[np.arange(batch_size)] = np.arange(num_time_periods)
        year_indices_rep = np.repeat(year_indices, num_time_periods, axis=1)
        time_period_input = batch_indices <= year_indices_rep
        time_period_input = time_period_input.astype(np.float32)
        return time_period_input

    def _vectorize_batch_text(self, batch_text, max_words):
        batch_words = [self._tokenizer(text) for text in batch_text]
        batch_word_indices = [[self._word_to_index(word) for word in words] for words in batch_words]
        vectorized_text = pad_sequences(batch_word_indices, maxlen=max_words, dtype='int32', padding='post', truncating='post', value=self._pp_config.padding_index)
        return vectorized_text

    def _word_to_index(self, word):
        index = self._word_index_lookup[word] if word in self._word_index_lookup else self._pp_config.unknown_word_index
        return index
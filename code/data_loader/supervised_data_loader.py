"""

"""

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import sys

import numpy as np

from utils.config_utils import get_list_of_files, compute_max_sent_len, open_file
from keras.utils.np_utils import to_categorical
from utils.data_utils import number_of_samples, apply_transformer, convert2indices
from utils.parse_utils import Preprocessing


senti_map = {
    'negative': 0,
    'positive': 1,
    'neutral': 2
}

lang_map = {
    'de': 0,
    'it': 1,
    'fr': 2,
    'en': 3,
}


class SupervisedDataLoader(object):
    def __init__(self, config_data, embeddings, mode='train'):
        self.type = 'loader'
        # files: (fname, file_type, tags, text_type, max_sentneces, max_idx)
        files = get_list_of_files(config_data['input_{}_directories'.format(mode)])

        max_sentence_len = config_data['max_sentence_length']

        if max_sentence_len == -1:
            self.max_sentence_len = 140
        else:
            self.max_sentence_len = max_sentence_len

        types = set(map(lambda x: x['text_type'], files))
        preprocessor = Preprocessing(types)
        vocabulary = embeddings.vocabulary
        transformers = embeddings.transformers

        if not mode == 'test':
            input= []
            senti = []
            language = []
            for train_file in files:
                preprocessed_data = self.transform_data(train_file, preprocessor, vocabulary, transformers, max_sentence_len)
                input.append(preprocessed_data['text'])
                senti.append(preprocessed_data['sentiment_label'])

            input_train_matrix = np.vstack(input)
            senti_train_matrix = np.hstack(senti)

            # join validation and training sets into one train set and one validation set
            self.input_data = [input_train_matrix]
            self.output_data = [to_categorical(senti_train_matrix, len(senti_map))]
            self.nsamples = input_train_matrix.shape[0]

        else:
            input_test = []
            senti_test = []
            language_test = []
            names_test = []
            ids_test = []
            for test_file in files:
                preprocessed_data = self.transform_data(test_file, preprocessor, vocabulary, transformers, max_sentence_len)
                input_test.append(preprocessed_data['text'])
                senti_test.append(to_categorical(preprocessed_data['sentiment_label'], len(senti_map)))
                language_test.append(to_categorical(preprocessed_data['language_label'], len(lang_map)))
                names_test.append(test_file['file_name'])
                ids_test.append(preprocessed_data['id'])

            #test sets are treated seperately
            self.input_data = input_test
            self.output_data = list(zip(senti_test, language_test))
            self.names = names_test
            self.ids = ids_test

    def transform_data(self, f, preprocessor, vocabulary, transformers, max_sentence_len):
        file = open_file(f['file_name'], f['file_type'])
        text_type = f['text_type']
        attributes = f['tags']
        max_index = f['max_index']
        if sys.version_info[0] < 3:
            curr_tweets = list(map(lambda x: x.decode('utf-8').replace('\n', '').split('\t'), file.readlines()))
        else:
            curr_tweets = map(lambda x: x.replace('\r', '').replace('\n', '').split('\t'), file.readlines())
        curr_tweets = list(filter(lambda x: len(x) == max_index + 1, curr_tweets))
        unk_word_idx = vocabulary['UNK'][0]
        dummy_word_idx = vocabulary['DUMMY_WORD'][0]

        preprocessed_data = {}
        for attribute in attributes:
            if attribute == 'text':
                text = map(lambda x: preprocessor.preprocess_text(x[attributes[attribute]], text_type), curr_tweets)
                text = list(map(lambda x: apply_transformer(x, transformers), text))
                text_idx = convert2indices(text, vocabulary, dummy_word_idx, unk_word_idx, max_sent_length=max_sentence_len)
                preprocessed_data[attribute] = text_idx
            elif attribute == 'sentiment_label':
                sentiments = list(map(lambda x: senti_map.get(x[attributes[attribute]], 1), curr_tweets))
                preprocessed_data[attribute] = np.asarray(sentiments)
            elif attribute == 'language_label':
                languages = list(map(lambda x: lang_map.get(x[attributes[attribute]]), curr_tweets))
                preprocessed_data[attribute] = np.asarray(languages)
            elif attribute == 'id':
                ids = list(map(lambda x: x[attributes[attribute]], curr_tweets))
                preprocessed_data[attribute] = np.asarray(ids)

        return preprocessed_data

    def max_sentence_len(self, config_data):
        tags = ['input_train_directories', 'input_valid_directories', 'input_test_directories']
        return max(map(lambda x: compute_max_sent_len(config_data[x]), tags))

    def number_of_samples(self, config_data):
        tags = ['input_train_directories', 'input_valid_directories', 'input_test_directories']
        return sum(map(lambda x: number_of_samples(config_data[x]), tags))

if __name__ == '__main__':
    with open('../configurations/config_distant_phase_it_test.json', 'r') as json_data:
        config_data = json.load(json_data)
        data_iterator = SupervisedDataLoader(config_data, mode='train')

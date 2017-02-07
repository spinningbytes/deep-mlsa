"""
This Dataloader is an iterator for long files, mostly used for sentiment&language detection task.
If multiple files are used they are processed in parallel, meaning that they are read batchwise and the batches are mixed into one batch
where the contents of the files is mixed.
"""

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import sys
from itertools import islice

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


class DistantDataParseIterator(object):
    def __init__(self, config_data, embeddings, mode='train'):
        self.type = 'generator'
        train_dirs = config_data['input_{}_directories'.format(mode)]
        # files: (fname, file_type, tags, text_type, max_sentneces, max_idx)
        self.files = get_list_of_files(train_dirs)
        max_sentence_len = compute_max_sent_len(train_dirs)
        if max_sentence_len == -1:
            self.max_sentence_len = 140
        else:
            self.max_sentence_len = max_sentence_len

        self.types = set(map(lambda x: x['text_type'], self.files))
        self.preprocessor = Preprocessing(self.types)
        self.vocabulary = embeddings.vocabulary
        self.transformers = embeddings.transformers
        self.nsamples = number_of_samples(train_dirs)

        if self.nsamples < 0:
            for f in self.files:
                self.nsamples += f['max_sentences']

        self.nsamples = self.nsamples/config_data['nb_epochs']

    def flow(self, batch_size=1000):
        it_batchsize = int(batch_size/len(self.files))
        #never ending iterator
        while True:
            iterators = list(map(lambda x: SingleFileBatchIterator(x, it_batchsize), self.files))
            while True:
                if len(iterators) == 0:
                    break
                tweets, senti_labels = [], []

                for iterator in iterators:
                    itbatch = iterator.get_next_batch()
                    if not itbatch:
                        iterators.remove(iterator)
                        continue
                    text_idx = iterator.text_idx
                    text_type = iterator.text_type
                    sentiment_label_idx = iterator.sentiment_label_idx
                    itweeets, isentiments = self.process_batch(itbatch, text_idx, text_type, sentiment_label_idx)

                    tweets.append(itweeets)
                    senti_labels.append(isentiments)

                assert len(tweets) == len(senti_labels)
                if len(tweets) == 0:
                    break

                rtweets = np.vstack(tweets)
                rsenti = np.vstack(senti_labels)

                p = np.random.permutation(len(rtweets))
                yield rtweets[p], rsenti[p]

    def process_batch(self, batch, text_idx, text_type, sentiment_label_idx):
        unk_word_idx = self.vocabulary['UNK'][0]
        dummy_word_idx = self.vocabulary['DUMMY_WORD'][0]
        if sys.version_info[0] < 3:
            curr_tweets = list(map(lambda x: x.decode('utf-8').replace('\n', '').split('\t'), batch))
        else:
            curr_tweets = list(map(lambda x: x.replace('\n', '').split('\t'), batch))
        tweets = map(lambda x: self.preprocessor.preprocess_text(x[text_idx], text_type), curr_tweets)
        tweets = list(map(lambda x: apply_transformer(x, self.transformers), tweets))
        tweet_idx = convert2indices(tweets, self.vocabulary, dummy_word_idx, unk_word_idx, max_sent_length=self.max_sentence_len)
        sentiments = np.asarray(list(map(lambda x: senti_map.get(x[sentiment_label_idx]), curr_tweets))).astype('int32')

        return tweet_idx, to_categorical(sentiments, len(senti_map))


class SingleFileBatchIterator(object):
    def __init__(self, ifname, batch_size=1000):
        self.sentences_left = ifname['max_sentences']
        self.text_type = ifname['text_type']
        self.max_index = ifname['max_index']
        self.ifile = open_file(ifname['file_name'], ifname['file_type'])
        attributes = ifname['tags']
        self.sentiment_label_idx = attributes['sentiment_label']
        self.text_idx = attributes['text']
        self.batch_size = batch_size

    def get_next_batch(self):
        if self.sentences_left <= 0:
            self.ifile.close()
            return []
        batch = []
        for line in islice(self.ifile, self.batch_size):
            if self.sentences_left <= 0:
                self.ifile.close()
                return batch
            batch.append(line)
            self.sentences_left -= 1

        return batch


if __name__ == '__main__':
    with open('../configurations/config_distant_phase_it_test.json', 'r') as json_data:
        config_data = json.load(json_data)
        data_iterator = DistantDataParseIterator(config_data)
        for tweet, (sentments, lengauges) in data_iterator.flow(batch_size=100):
            print(tweet.shape)
            print(lengauges)

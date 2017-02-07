"""
This Dataloader is an iterator for long files, mostly used for sentiment&language detection task.
If multiple files are used they are processed sequencially. Look at the mixed_iterator if the file content needs to be mixed.
This is expecially useful if the Task is based only on a single large file.
"""

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np

from utils.config_utils import get_list_of_files, compute_max_sent_len, open_file
from keras.utils.np_utils import to_categorical
from utils.data_utils import number_of_samples, apply_transformer, convert2indices
from utils.parse_utils import Preprocessing
import sys

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

        self.nsamples = self.nsamples / config_data['nb_epochs']

    def process_batch(self, batch, lang_tag, text_idx, text_type, sentiment_label_idx):
        unk_word_idx = self.vocabulary['UNK'][0]
        dummy_word_idx = self.vocabulary['DUMMY_WORD'][0]
        if sys.version_info[0] < 3:
            curr_tweets = list(map(lambda x: x.decode('utf-8').replace('\n', '').split('\t'), batch))
        else:
            curr_tweets = list(map(lambda x: x.replace('\n', '').split('\t'), batch))

        curr_tweets = list(map(lambda x: x.replace('\n', '').split('\t'), curr_tweets))
        lang = list(map(lambda x: lang_map.get(x[lang_tag]), curr_tweets))
        languages = np.asarray(lang)
        tweets = map(lambda x, y: self.preprocessor.preprocess_text(x[text_idx], text_type, language=y), curr_tweets, languages)
        tweets = list(map(lambda x: apply_transformer(x, self.transformers), tweets))
        tweet_idx = convert2indices(tweets, self.vocabulary, dummy_word_idx, unk_word_idx,max_sent_length=self.max_sentence_len)
        sentiments = np.asarray(list(map(lambda x: senti_map.get(x[sentiment_label_idx]), curr_tweets)))

        return tweet_idx, to_categorical(sentiments, 3)

    def flow(self, batch_size=1000):
        unk_word_idx = self.vocabulary['UNK'][0]
        dummy_word_idx = self.vocabulary['DUMMY_WORD'][0]

        while True:
            for f in self.files:
                max_sentences = f['max_sentences']
                if max_sentences == 0:
                    continue
                file = open_file(f['file_name'], f['file_type'])
                text_type = f['text_type']
                attributes = f['tags']
                max_index = f['max_index']
                curr_tweets = []

                text_idx = attributes['text']
                sentiment_label_idx = attributes['sentiment_label']

                for i, line in enumerate(file):
                    curr_tweets.append(line)
                    if len(curr_tweets) == batch_size or i >= max_sentences:
                        if sys.version_info[0] < 3:
                            curr_tweets = list(map(lambda x: x.decode('utf-8').replace('\n', '').split('\t'), curr_tweets))
                        else:
                            curr_tweets = list(map(lambda x: x.replace('\n', '').split('\t'), curr_tweets))

                        tweets = map(lambda x: self.preprocessor.preprocess_text(x[text_idx], text_type), curr_tweets)
                        tweets = list(map(lambda x: apply_transformer(x, self.transformers), tweets))
                        tweet_idx = convert2indices(tweets, self.vocabulary, dummy_word_idx, unk_word_idx, max_sent_length=self.max_sentence_len)

                        sentiments = np.asarray(list(map(lambda x: senti_map.get(x[sentiment_label_idx]), curr_tweets)))

                        yield tweet_idx, to_categorical(sentiments, 3)
                        curr_tweets = []
                    if i >= max_sentences:
                        break

                file.close()

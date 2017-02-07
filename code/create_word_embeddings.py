# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import getopt
import json
import logging
import os
import sys
from itertools import islice

import _pickle as cPickle
import numpy as np
from gensim.models import Word2Vec, Phrases

from utils.config_utils import get_list_of_files, open_file
from utils.parse_utils import Preprocessing


class SequencialIterator(object):
    def __init__(self, files):
        self.files = files
        types = map(lambda x: x['text_type'], files)
        self.preprocessor = Preprocessing(types)

    def __iter__(self):
        for f in self.files:
            max_sentences = f['max_sentences']
            file = open_file(f['file_name'], f['file_type'])
            text_type = f['text_type']
            text_idx = f['tags'].values()
            max_index = f['max_index']
            for i, line in enumerate(file):
                if i >= max_sentences:
                    break
                line = line.replace('\r', '').replace('\n', '').split('\t')
                if len(line) == max_index + 1:
                    for idx in text_idx:
                        text = self.preprocessor.preprocess_text(line[idx], text_type)
                        yield text


class MixedIterator(object):
    def __init__(self, files, batch_size=10000):
        self.files = files
        self.types = set(map(lambda x: x['text_type'], files))
        self.batch_size = batch_size

    def __iter__(self):
        iterators = list(map(lambda x: SingleFileBatchIterator(x, self.types, self.batch_size), self.files))
        while True:
            if len(iterators) == 0:
                break
            for iterator in iterators:
                it_batch = iterator.get_next_batch()
                if not it_batch:
                    iterators.remove(iterator)
                    continue
                for line in it_batch:
                    yield line


class SingleFileBatchIterator(object):
    def __init__(self, ifname, types, batch_size=10000):
        self.max_sentences = ifname['max_sentences']
        self.text_type = ifname['text_type']
        self.text_idx = ifname['tags'].values()
        self.max_index = ifname['max_index']
        self.ifile = open_file(ifname['file_name'], ifname['file_type'])
        self.batch_size = batch_size
        self.preprocessor = Preprocessing(types)

    def get_next_batch(self):
        if self.max_sentences <= 0:
            self.ifile.close()
            return []
        batch = []
        for line in islice(self.ifile, self.batch_size):
            line = line.replace('\r', '').replace('\n', '').split('\t')
            if self.max_sentences <= 0:
                self.ifile.close()
                return batch
            for idx in self.text_idx:
                line_preprocessed = self.preprocessor.preprocess_text(line[idx], text_type=self.text_type)
                batch.append(line_preprocessed)
            self.max_sentences -= 1
        return batch


def copy_vocabulary(vocab):
    vocabulary = {}
    max_idx = 0
    for (k, v) in vocab.items():
        vocabulary[k] = (v.index, v.count)
        max_idx = max([v.index, max_idx])

    #add an unknown word placeholder and dummyword (used for padding)
    vocabulary['UNK'] = (max_idx + 1, 1)
    vocabulary['DUMMY_WORD'] = (max_idx + 2, 1)
    return vocabulary


def main(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    try:
        opts, args = getopt.getopt(args, "c:", ["config="])
    except getopt.GetoptError:
        print('usage: -c my_config.json')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c", "--config"):
            config_fname = os.path.join('configurations', arg)

    with open(config_fname, 'r') as json_data:
        config_data = json.load(json_data)
        word2vec_params = config_data['word2vec_parameter']
        max_vocab_size = word2vec_params['max_vocab_size']
        files = get_list_of_files(config_data['input_directories'])
        if config_data['sequence'] == 'mixed':
            sentences = MixedIterator(files, batch_size=1000)
        else:
            sentences = SequencialIterator(files)

        #prepare all the word2phrase transformers, first unigram to bigram, then bigram to 3-gram,..
        ngram_processors = []
        for i in range(config_data['ngrams'] - 1):
            transformed_sentences = sentences
            for ngram_processor in ngram_processors:
                transformed_sentences = ngram_processor[transformed_sentences]

            bigram_trainsformer = Phrases(transformed_sentences, min_count=word2vec_params['min_count'], max_vocab_size=max_vocab_size)
            ngram_processors.append(bigram_trainsformer)

        transformed_sentences = sentences
        for ngram_processor in ngram_processors:
            transformed_sentences = ngram_processor[transformed_sentences]

        #train word embeddings

        model = Word2Vec(
            transformed_sentences,
            size=word2vec_params['size'],
            window=word2vec_params['window'],
            min_count=word2vec_params['min_count'],
            workers=word2vec_params['workers'],
            sg=word2vec_params['sg'],
            sample=word2vec_params['sample'],
            hs=word2vec_params['hs'],
            iter=word2vec_params['iter'],
            negative=word2vec_params['negative'],
            max_vocab_size=max_vocab_size
        )

        #save all the data
        output_basename = config_data['output_basename']
        output_dir = os.path.join(config_data['output_directory'], output_basename)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        model.save_word2vec_format(os.path.join(output_dir, 'embedding_file'), binary=False)
        vocabulary = copy_vocabulary(model.vocab)

        #create numpy embedding matrix as input for the EmbeddingLayer in Keras
        vocab_emb = np.zeros((len(vocabulary) + 1, word2vec_params['size']), dtype='float32')
        for word, (idx, freq) in vocabulary.items():
            word_vec = None
            if word in model.vocab.keys():
                word_vec = model[word]
            if word_vec is None:
                word_vec = np.random.uniform(-0.25, 0.25, word2vec_params['size'])
            vocab_emb[idx] = word_vec

        outfile = os.path.join(output_dir, 'embedding_matrix.npy')
        np.save(outfile, vocab_emb)

        #save the vocabulary, which is augmented by inserting an unknown word and dummy word
        cPickle.dump(vocabulary, open(os.path.join(output_dir, 'vocabulary.pickle'), 'wb'))
        json.dump(config_data, open(os.path.join(output_dir, 'config.json'), 'wt'))

        #save word2vec transformers in order of their application
        for i, ngram_processor in enumerate(ngram_processors):
            ngram_processor.save(os.path.join(output_dir, '{}gram'.format(i)))

if __name__ == '__main__':
    main(sys.argv[1:])
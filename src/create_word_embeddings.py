# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import getopt
import json
import logging
import os
import sys

import argparse
import _pickle as cPickle
import numpy as np
from gensim.models import Word2Vec, Phrases

from src.unlabelled_data_loader.data_generator import DataGenerator


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
    parser = argparse.ArgumentParser(description='Create Word Embeddings')
    parser.add_argument('-c, --configuration', dest='config', type=str, help='The configuration file name which needs to be stored in configurations/mlsa_create_wordembeddings')

    args = parser.parse_args(args=args)
    config = args.config
    config_fname = os.path.join('configurations/mlsa_create_wordembeddings', config)

    with open(config_fname, 'r') as json_data:
        config_data = json.load(json_data)
        word2vec_params = config_data['word2vec_parameter']
        max_vocab_size = word2vec_params['max_vocab_size']
        language = config_data['language']
        file_name = config_data['file_name']
        nsentences = config_data['nsentences'] if config_data['nsentences'] > 0 else np.inf

        sentences = DataGenerator(file_name, config_data, language, nsentences)

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

        model.wv.save_word2vec_format(os.path.join(output_dir, 'embedding_file'), binary=False)
        vocabulary = copy_vocabulary(model.wv.vocab)

        #create numpy embedding matrix as input for the EmbeddingLayer in Keras
        vocab_emb = np.zeros((len(vocabulary) + 1, word2vec_params['size']), dtype='float32')
        for word, (idx, freq) in vocabulary.items():
            word_vec = None
            if word in model.wv.vocab.keys():
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
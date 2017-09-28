from nltk.tokenize import TweetTokenizer, WordPunctTokenizer
import numpy as np
import os
import re


tokenizers = {
    'en': TweetTokenizer(),
    'de': WordPunctTokenizer(),
    'it': WordPunctTokenizer(),
    'fr': WordPunctTokenizer(),
    'default': WordPunctTokenizer()
}

tokenizer = TweetTokenizer()


def preprocess(tweet, lang):
    #lowercase and normalize urls
    tweet = tweet.lower()
    tweet = tweet.replace('\n', '')
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', 'URLTOK', tweet)
    tweet = re.sub('@[^\s]+', 'USRTOK', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    tweet = tokenizers.get(lang, tokenizer).tokenize(tweet)

    return list(map(lambda x: x.replace(' ', ''), tweet))


def apply_transformer(sentence, transofrmers):
    for transofrmer in transofrmers:
        sentence = transofrmer[sentence]
    return sentence


def number_of_samples(directory_list):
    nsamples = 0
    for directory in directory_list:
        nsamples += directory['max_sentences']
    return nsamples


def load_embedding_matrix(config_data):
    path = config_data['embeddings']
    matrix_fname = os.path.join(path, 'embedding_matrix.npy')
    embedding_matrix = np.load(matrix_fname)
    return embedding_matrix


def convert2indices(data, alphabet, dummy_word_idx, unk_word_idx, max_sent_length=140, verbose=0):
    data_idx = []
    max_len = 0
    unknown_words = 0
    known_words = 0
    for sentence in data:
        ex = np.ones(max_sent_length) * dummy_word_idx
        max_len = max(len(sentence), max_len)
        if len(sentence) > max_sent_length:
            sentence = sentence[:max_sent_length]
        for i, token in enumerate(sentence):
            idx, freq = alphabet.get(token, (unk_word_idx, 0))
            ex[i] = idx
            if idx == unk_word_idx:
                unknown_words += 1
            else:
                known_words += 1
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('int32')
    if verbose == 1:
        print("Max length in this batch:", max_len)
        print("Number of unknown words:", unknown_words)
        print("Number of known words:", known_words)
    return data_idx
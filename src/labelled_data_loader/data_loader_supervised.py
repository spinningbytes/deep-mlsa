import numpy as np
from keras.utils.np_utils import to_categorical
from src.utils.data_utils import  apply_transformer, convert2indices
from src.utils.parse_utils import preprocess

#set neg pos to 0 and 1 so that it can be reaused for binary classification
senti_map = {
    'negative': 0,
    'positive': 1,
    'neutral': 2
}


def load_data(filename, config_data, vocabulary, transformers):
    max_sentence_length = config_data['max_sentence_length']
    dummy_word_idx = vocabulary['DUMMY_WORD'][0]
    unk_word_idx = vocabulary['UNK'][0]

    tid_idx = config_data['tid_idx']
    text_idx = config_data['text_idx']
    label_idx = config_data['label_idx']
    language = config_data['language']

    ifile = open(filename, encoding='utf-8', mode='rt')
    split_data = [x.replace('\r', '').split('\t') for x in ifile.readlines()]

    #set everything that is not pos or negative to netural
    tids = [x[tid_idx] for x in split_data]
    labels = to_categorical(np.asarray([senti_map.get(x[label_idx], 2) for x in split_data]))
    preprocessed_tweets = [preprocess(x[text_idx], lang=language) for x in split_data]
    #apply ngrams
    phrased_tweets = [apply_transformer(x, transformers) for x in preprocessed_tweets]

    text_to_idx = convert2indices(phrased_tweets, vocabulary, dummy_word_idx, unk_word_idx, max_sent_length=max_sentence_length)

    return tids, labels, text_to_idx
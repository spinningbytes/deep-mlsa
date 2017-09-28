import numpy as np
import gzip
from keras.utils.np_utils import to_categorical
from src.utils.data_utils import apply_transformer, convert2indices
import random
from src.utils.parse_utils import preprocess


senti_map = {
    'negative': 0,
    'positive': 1,
    'neutral': 2
}


def generate_data(filename, config_data, vocabulary, transformers, language, batch_size):
    max_sentence_length = config_data['max_sentence_length']
    dummy_word_idx = vocabulary['DUMMY_WORD'][0]
    unk_word_idx = vocabulary['UNK'][0]

    text_idx = config_data['text_idx']
    label_idx = config_data['label_idx']

    current_batch = []
    while True:
        ifile = gzip.open(filename, mode='rb')
        for i, line in enumerate(ifile, start=1):
            current_batch.append(line)
            if i % batch_size == 0:
                random.shuffle(current_batch)
                split_data = [x.replace('\r', '').split('\t') for x in current_batch]
                labels = to_categorical(np.asarray([senti_map.get(x[label_idx], 2) for x in split_data]))
                preprocessed_tweets = [preprocess(x[text_idx], lang=language) for x in split_data]
                phrased_tweets = [apply_transformer(x, transformers) for x in preprocessed_tweets]
                text_to_idx = convert2indices(phrased_tweets, vocabulary, dummy_word_idx, unk_word_idx,max_sentence_length)

                yield text_to_idx, labels
                current_batch = []

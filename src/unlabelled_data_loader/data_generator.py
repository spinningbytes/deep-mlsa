import json
import numpy as np
import gzip
from src.utils.parse_utils import preprocess


class DataGenerator(object):
    def __init__(self, filename, config_data, language, nsentences=np.inf):
        self.filename = filename
        self.tet_idx = config_data['text_idx']
        self.language = language
        self.nsentences = nsentences

    def __iter__(self):
        ifile = gzip.open(self.filename, mode='rt', encoding='utf-8')
        for i, line in enumerate(ifile, start=1):
            if i >= self.nsentences:
                break

            split_line = line.replace('\r', '').split('\t')
            preprocessed_tweet = preprocess(split_line[self.tet_idx], lang=self.language)

            yield preprocessed_tweet


if __name__ == '__main__':
    config_fname = 'configurations/mlsa_create_wordembeddings/config_word_embeddings_ch.json'
    with open(config_fname, 'r') as json_data:
        config_data = json.load(json_data)
        gen = DataGenerator(config_data['file_name'], config_data, 'en', 10)

        for text in gen:
            print(text)

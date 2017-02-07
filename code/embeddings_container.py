import sys
import os
import json
if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle
from gensim.models import Phrases


class EmbeddingContainer(object):
    def __init__(self, config_data):
        self.vocabulary = self.load_vocabulary(config_data)
        self.transformers = self.load_transformer_list(config_data)

    @staticmethod
    def load_vocabulary(config_data):
        output_directory = config_data['embeddings_directory']
        output_basename = config_data['embeddings_basename']

        path = os.path.join(output_directory, output_basename)
        fname = os.path.join(path, 'vocabulary.pickle')

        vocabulary = cPickle.load(open(fname, 'rb'))
        return vocabulary

    @staticmethod
    def load_transformer_list(config_data):
        output_directory = config_data['embeddings_directory']
        output_basename = config_data['embeddings_basename']

        path = os.path.join(output_directory, output_basename)
        config_fname = os.path.join(path, 'config.json')
        with open(config_fname, 'r') as json_data:
            wemb_config = json.load(json_data)
            ngrams = wemb_config['ngrams']

            transformers = []
            for i in range(ngrams - 1):
                phrase_model = Phrases.load(os.path.join(path, '{}gram'.format(i)))
                transformers.append(phrase_model)

        return transformers
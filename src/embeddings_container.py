import sys
import os
import json
if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle
from gensim.models import Phrases



def load_vocabulary(config_data):
    path = config_data['embeddings']
    fname = os.path.join(path, 'vocabulary.pickle')

    vocabulary = cPickle.load(open(fname, 'rb'))
    return vocabulary

def load_transformer_list(config_data):
    path = config_data['embeddings']
    config_fname = os.path.join(path, 'config.json')
    with open(config_fname, 'r') as json_data:
        wemb_config = json.load(json_data)
        ngrams = wemb_config['ngrams']

        transformers = []
        for i in range(ngrams - 1):
            phrase_model = Phrases.load(os.path.join(path, '{}gram'.format(i)))
            transformers.append(phrase_model)

    return transformers

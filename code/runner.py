# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import json
import logging
import sys
import os
import getopt

from embeddings_container import EmbeddingContainer
from utils import run_utils, fit_utils, model_utils
import numpy as np
from keras.models import model_from_json


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
        np.random.seed(1337)

        path = config_data['output_path']
        basename = config_data['output_basename']
        base_path = os.path.join(path, basename)

        embeddings = EmbeddingContainer(config_data)
        train_iterator = None
        valid_iterator = None
        test_iterator = None

        if config_data['input_train_directories']:
            logging.info('Loading Training Data')
            train_iterator = run_utils.get_iterator(config_data, embeddings, mode='train')

        if config_data['input_valid_directories']:
            logging.info('Loading Validation Data')
            valid_iterator = run_utils.get_iterator(config_data, embeddings, mode='valid')
        else:
            valid_iterator = None

        if config_data['input_test_directories']:
            logging.info('Loading Testing Data')
            test_iterator = run_utils.get_iterator(config_data, embeddings, mode='test')

        if train_iterator is not None:
            model, test_model = run_utils.get_model(config_data, train_iterator)
            json_string = model.to_json()
            open(os.path.join(base_path, 'model_definition.json'), 'wt').write(json_string)
            json_string = test_model.to_json()
            open(os.path.join(base_path, 'test_model_definition.json'), 'wt').write(json_string)
        else:
            json_string = open(os.path.join(base_path, 'model_definition.json'), 'rt').read()
            model = model_from_json(json_string)

            json_string = open(os.path.join(base_path, 'test_model_definition.json'), 'rt').read()
            test_model = model_from_json(json_string)

        model.summary()
        if not config_data['pretrained_model_directory'] == 'None' and not config_data['pretrained_model_basename'] == 'None':
            pretrained_model_path = os.path.join(config_data['pretrained_model_directory'], config_data['pretrained_model_basename'])
            pretrained_model = os.path.join(pretrained_model_path, 'best_model.h5')
            logging.info('Loading Pretraind Model from: {}'.format(pretrained_model))
            model.load_weights(pretrained_model)
            if config_data.get('transfer_learning', None) == 'True':
                model = model_utils.pop_layer(model, config_data['nlabels'])

        optimizer = run_utils.get_optimizer(config_data)
        metrics = run_utils.get_evaluation_metric(config_data)

        model.compile(loss=config_data['loss'], optimizer=optimizer, metrics=metrics)

        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(base_path):
            os.mkdir(base_path)

        history, stored_model, appendices = fit_utils.fit_model(config_data, model, train_iterator, valid_iterator)

        if test_iterator:
            weights_path = os.path.join(base_path, 'best_model.h5')
            if not (stored_model or config_data['nb_epochs'] == 0):
                logging.info('Storing Trained Model')
                model.save_weights(weights_path, overwrite=True)

            logging.info('Load Trained Model')
            test_model.load_weights(weights_path)
            json.dump(config_data, open(os.path.join(base_path, 'config.json'), 'w'))

            result_path = os.path.join('results', basename)
            if not os.path.exists(result_path):
                os.mkdir(result_path)

            if history is not None:
                for h in history.history.keys():
                    n = history.history[h]
                    np.save(open(os.path.join(result_path, '{}.npy'.format(h)), 'wb'), n)

            oline_test = run_utils.get_evaluation(config_data, test_model, test_iterator, basename, result_path)
            print(oline_test)
            open(os.path.join('results', 'results_log.tsv'), 'at').write(oline_test + '\n')

if __name__ == '__main__':
    main(sys.argv[1:])
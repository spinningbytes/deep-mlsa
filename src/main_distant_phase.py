import json
import logging
import sys
import os
import argparse
from math import ceil

from src.embeddings_container import load_transformer_list, load_vocabulary
from src.architectures.default_cnn import create_default_model
from src.labelled_data_loader.data_generator import generate_data
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
from src.evaluation_metrics.evaluation_metrics_theano import f1_score_semeval
from keras.models import model_from_json


def main(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Create Word Embeddings')
    parser.add_argument('-c, --configuration', dest='config', type=str,
                        help='The configuration file name which needs to be stored in configurations/mlsa_create_wordembeddings')

    args = parser.parse_args(args=args)
    config = args.config
    config_fname = os.path.join('configurations/mlsa_train_distant_phase', config)

    with open(config_fname, 'r') as json_data:
        config_data = json.load(json_data)
        np.random.seed(1337)

        input_file = config_data['input_file']
        vocabulary = load_vocabulary(config_data)
        transformers = load_transformer_list(config_data)
        batch_size = config_data['batch_size']
        language = config_data['language']
        output_path = config_data['output_path']
        samples_per_epoch = config_data['samples_per_epoch']
        nb_epochs = config_data['nb_epochs']
        nsamples = config_data['nsamples']

        model = create_default_model(config_data, nlabels=2)
        model.summary()

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        model_checkpoint = ModelCheckpoint(os.path.join(output_path, 'weights.{epoch:02d}.hdf5'), period=10, save_weights_only=True)

        steps_per_epoch = ceil(samples_per_epoch / batch_size)
        epochs = ceil(nb_epochs * (nsamples / samples_per_epoch))

        adadelta = Adadelta(lr=1.0, epsilon=1e-6, rho=0.95)
        model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=[f1_score_semeval, 'accuracy'])

        model.fit_generator(
            generator=generate_data(input_file, config_data, vocabulary, transformers, language, batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[model_checkpoint]
        )


if __name__ == '__main__':
    main(sys.argv[1:])





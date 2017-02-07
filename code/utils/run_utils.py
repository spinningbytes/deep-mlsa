# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os

from data_loader.distant_phase_mixed_iterator import DistantDataParseIterator
from data_loader.supervised_data_loader import SupervisedDataLoader
from data_loader.pair_data_iterator import PairDataIterator
from data_loader.cqa_data_loader import CQADataLoader
from data_loader.sts_data_loader import STSDataLoader
from data_loader.cqa_data_generator import CQADataGenerator
import data_loader.distant_phase_sequencial_iterator
from architectures.default_cnn import create_default_model
from architectures.abcnn import build_model_abcnn
from architectures.abcnn2 import build_model_abcnn2
from architectures.abcnn3 import build_model_abcnn3
from architectures.abcnn4 import build_model_abcnn4
from architectures.abcnn5 import build_model_abcnn5
from architectures.bcnn import build_model_bcnn
from architectures.abnn import build_model_abnn
from architectures.abcnn_psenti import build_model_abcnn_psenti
from architectures.abcnn_wnet import build_model_abcnn_wnet

from evaluation.senti_lang_eval import evaluate
from evaluation.pair_sentiment_eval import evaluate_pair
from evaluation import cqa_eval
from evaluation import sts_eval
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import evaluation_metrics
import keras.optimizers as optimizers


def get_iterator(config_data, embeddings, mode='train'):
    if config_data['{}_loader_type'.format(mode)] == 'supervised_data_loader':
        return SupervisedDataLoader(config_data, embeddings, mode=mode)
    elif config_data['{}_loader_type'.format(mode)] == 'distant_phase_mixed_iterator':
        return DistantDataParseIterator(config_data, embeddings)
    elif config_data['{}_loader_type'.format(mode)] == 'distant_phase_sequencial_iterator':
        return data_loader.distant_phase_sequencial_iterator.DistantDataParseIterator(config_data, embeddings)
    elif config_data['{}_loader_type'.format(mode)] == 'pair_data_iterator':
        return PairDataIterator(config_data, embeddings, mode=mode)
    elif config_data['{}_loader_type'.format(mode)] == 'cqa_data_loader':
        return CQADataLoader(config_data, embeddings, mode=mode)
    elif config_data['{}_loader_type'.format(mode)] == 'sts_data_loader':
        return STSDataLoader(config_data, embeddings, mode=mode)
    elif config_data['{}_loader_type'.format(mode)] == 'cqa_data_generator':
        return CQADataGenerator(config_data, embeddings, mode=mode)
    else:
        return None


def get_model(config_data, train_iterator):
    if config_data['model'] == 'default_cnn':
        return create_default_model(config_data, train_iterator.max_sentence_len)
    elif config_data['model'] == 'abcnn':
        return build_model_abcnn(config_data, train_iterator.max_sent_len0, train_iterator.max_sent_len1)
    elif config_data['model'] == 'abcnn2':
        return build_model_abcnn2(config_data, train_iterator.max_sent_len0, train_iterator.max_sent_len1)
    elif config_data['model'] == 'abcnn3':
        return build_model_abcnn3(config_data, train_iterator.max_sent_len0, train_iterator.max_sent_len1)
    elif config_data['model'] == 'abcnn4':
        return build_model_abcnn4(config_data, train_iterator.max_sent_len0, train_iterator.max_sent_len1)
    elif config_data['model'] == 'bcnn':
        return build_model_bcnn(config_data, train_iterator.max_sent_len0, train_iterator.max_sent_len1)
    elif config_data['model'] == 'abcnn5':
        return build_model_abcnn5(config_data, train_iterator.max_sent_len0, train_iterator.max_sent_len1)
    elif config_data['model'] == 'abnn':
        return build_model_abnn(config_data, train_iterator.max_sent_len0, train_iterator.max_sent_len1)
    elif config_data['model'] == 'abcnn_psenti':
        return build_model_abcnn_psenti(config_data, train_iterator.max_sent_len0, train_iterator.max_sent_len1)
    elif config_data['model'] == 'abcnn_wnet':
        return build_model_abcnn_wnet(config_data, train_iterator.max_sent_len0, train_iterator.max_sent_len1)
    else:
        return None


def get_callbacks(config_data, appendix=''):
    ret_callbacks = []
    model_stored = False
    callbacks = config_data['callbacks']
    tensor_board = TensorBoard(log_dir=os.path.join('logging', config_data['tb_log_dir']), histogram_freq=10)
    ret_callbacks.append(tensor_board)
    for callback in callbacks:
        if callback['name'] == 'early_stopping':
            ret_callbacks.append(EarlyStopping(monitor=callback['monitor'], patience=callback['patience'], verbose=callback['verbose'], mode=callback['mode']))
        elif callback['name'] == 'model_checkpoit':
            model_stored = True
            path = config_data['output_path']
            basename = config_data['output_basename']
            base_path = os.path.join(path, basename)
            opath = os.path.join(base_path, 'best_model{}.h5'.format(appendix))
            save_best = bool(callback['save_best_only'])
            ret_callbacks.append(ModelCheckpoint(filepath=opath, verbose=callback['verbose'], save_best_only=save_best, monitor=callback['monitor'], mode=callback['mode']))
    return ret_callbacks, model_stored


def get_evaluation(config_data, model, test_iterator, basename, opath):
    ofname = os.path.join(opath, 'outfile.tsv')
    if config_data['evaluation'] == 'senti_lang_eval':
        return evaluate(model, test_iterator, basename, ofname)
    elif config_data['evaluation'] == 'pair_sentiment_eval':
        return evaluate_pair(config_data, model, test_iterator, basename, opath)
    elif config_data['evaluation'] == 'cqa_eval':
        return cqa_eval.evaluate(model, test_iterator, basename, opath)
    elif config_data['evaluation'] == 'sts_eval':
        return sts_eval.evaluate(model, test_iterator, basename, opath)
    else:
        return 'None'


def get_evaluation_metric(config_data):
    name = config_data['metrics']
    if name == 'f1_score_keras':
        return [evaluation_metrics.f1_score_keras]
    elif name == 'f1_score_semeval':
        return [evaluation_metrics.f1_score_semeval]
    elif name == 'f1_score_task3':
        return [evaluation_metrics.f1_score_task3]
    elif name == 'f05_score_task3':
        return [evaluation_metrics.f05_score_task3]
    elif name == 'precision_score_task3':
        return [evaluation_metrics.precision_score_task3]
    elif name == 'precision_keras':
        return [evaluation_metrics.precision_keras]
    elif name == 'accuracy':
        return ['accuracy']
    elif name == 'f1_score_taskB':
        return [evaluation_metrics.f1_score_taskB]
    elif name == 'pearsons_correlation':
        return [evaluation_metrics.pearsons_correlation]
    else:
        return []


def get_classweight(config_data):
    if config_data.get('class_weight', None):
        return {
            0: config_data['class_weight']['0'],
            1: config_data['class_weight']['1']
        }
    else:
        return None


def get_optimizer(config_data):
    options = config_data['optimizer']
    name = options['name']

    if name == 'adadelta':
        return optimizers.Adadelta(lr=options['lr'], rho=options['rho'], epsilon=options['epsilon'])
    else:
        return optimizers.SGD()
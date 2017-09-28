# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import json
import gzip
import os
import sys
import codecs


def get_list_of_files(directory_list):
    files = []
    for directory in directory_list:
        dir_name = directory['directory_name']

        schema_dir = directory['schema_directory']

        with open(os.path.join(schema_dir, 'schema.json'), 'r') as json_data:
            schema = json.load(json_data)
            schema_attributes = schema['attributes']
            schema_meta_data = schema['meta_data']

            #tsv, gzip,..
            file_type = schema_meta_data['file_type']

            #tweets, news, used to decide parsing
            text_type = schema_meta_data['text_type']

            #limit the number of sentence used, if limit is -1, then use all the data
            max_sentences = directory['max_sentences']
            if max_sentences == -1:
                max_sentences = schema_meta_data['length']

            #number of attributes in the file
            max_idx = max(filter(lambda x: type(x) == int, schema_attributes.values()))

        if not directory['file_names']:
            for fname in os.listdir(dir_name):
                if fname == 'schema.json':
                    continue
                files.append({
                    'file_name': os.path.join(dir_name, fname),
                    'file_type': file_type,
                    'tags': schema_attributes,
                    'text_type': text_type,
                    'max_sentences': max_sentences,
                    'max_index': max_idx
                })
        else:
            for fname in directory['file_names']:
                if fname == 'schema.json':
                    continue
                files.append({
                    'file_name': os.path.join(dir_name, fname),
                    'file_type': file_type,
                    'tags': schema_attributes,
                    'text_type': text_type,
                    'max_sentences': max_sentences,
                    'max_index': max_idx
                })
    return files


def open_file(fname, file_type):
    if file_type == 'gzip':
        if sys.version_info[0] < 3:
            return gzip.open(fname, mode='rt')
        else:
            return gzip.open(fname, mode='rt', encoding='utf-8')
    elif file_type == 'tsv':
        if sys.version_info[0] < 3:
            return codecs.open(fname,  mode='r', encoding='utf-8')
        else:
            return open(fname, encoding='utf-8', mode='rt')


def compute_max_sent_len(directory_list):
    max_sentence_length = -1
    for directory in directory_list:
        dir_name = directory['directory_name']

        with open(os.path.join(dir_name, 'schema.json'), 'r') as json_data:
            schema = json.load(json_data)
            schema_meta_data = schema['meta_data']
            sen_len = schema_meta_data['sentence_length']
            max_sentence_length = max([sen_len, max_sentence_length])

    return max_sentence_length


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False



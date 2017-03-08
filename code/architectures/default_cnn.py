import logging

from keras.layers import Dense, ZeroPadding1D, Embedding, Convolution1D, MaxPooling1D, Flatten, Input
from keras.models import Model
from utils.data_utils import load_embedding_matrix


def create_default_model(config_data):
    nb_filter = 200
    filter_length = 6
    hidden_dims = nb_filter

    embedding_matrix = load_embedding_matrix(config_data)
    max_features = embedding_matrix.shape[0]
    embedding_dims = embedding_matrix.shape[1]

    max_len = config_data['max_sentence_length']

    logging.info('Build Model...')
    logging.info('Embedding Dimensions: ({},{})'.format(max_features, embedding_dims))

    main_input = Input(batch_shape=(None, max_len), dtype='int32', name='main_input')
    if not config_data.get('random_embedding', None):
        logging.info('Pretrained Word Embeddings')
        embeddings = Embedding(
            max_features,
            embedding_dims,
            input_length=max_len,
            weights=[embedding_matrix],
            trainable=False
        )(main_input)
    else:
        logging.info('Random Word Embeddings')
        embeddings = Embedding(max_features, embedding_dims, init='lecun_uniform', input_length=max_len)(main_input)

    zeropadding = ZeroPadding1D(filter_length - 1)(embeddings)

    conv1 = Convolution1D(
        nb_filter=nb_filter,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1)(zeropadding)

    max_pooling1 = MaxPooling1D(pool_length=4, stride=2)(conv1)

    conv2 = Convolution1D(
        nb_filter=nb_filter,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1)(max_pooling1)

    max_pooling2 = MaxPooling1D(pool_length=conv2._keras_shape[1])(conv2)
    flatten = Flatten()(max_pooling2)
    hidden = Dense(hidden_dims)(flatten)
    softmax_layer1 = Dense(3, activation='softmax', name='sentiment_softmax', init='lecun_uniform')(hidden)

    model = Model(input=[main_input], output=softmax_layer1)

    test_model = Model(input=[main_input], output=[softmax_layer1, hidden])

    return model, test_model
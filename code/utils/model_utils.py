import keras.regularizers as regs
from keras.layers import Dense
from keras.models import Model

def pop_layer(model, nlabels):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers[-1].trainable = False
    softmax_layer1 = Dense(nlabels, activation='softmax', name='sentiment_softmax', init='lecun_uniform')(model.layers[-2].output)
    model = Model(input=model.inputs, output=softmax_layer1)
    return model

def get_regularizer(config_data):
    reg = config_data['regularizer']
    name = reg['name']
    if name == 'l2':
        return regs.l2(l=reg['l2'])
    else:
        return None

from data_loader.supervised_data_loader import SupervisedDataLoader
from keras.utils.np_utils import probas_to_classes
from sklearn.metrics import f1_score
import numpy as np


def evaluate(model, test_iteraotr, experiment_name, ofname):
    inputs = test_iteraotr.input_data
    outputs = test_iteraotr.output_data
    names = test_iteraotr.names
    ids_list = test_iteraotr.ids

    output_line = '{}\n'.format(experiment_name)
    for i, o, n, ids in zip(inputs, outputs, names, ids_list):
        y_test_senti = probas_to_classes(o)

        y_pred = model.predict(i)

        if type(y_pred) == list:
            y_pred_senti = y_pred[0]
        else:
            y_pred_senti = y_pred

        y_pred_senti_cls = probas_to_classes(y_pred_senti)

        f1_score_senti = f1_score(y_test_senti, y_pred_senti_cls, average=None, pos_label=None, labels=[0,1,2])

        output_line += '{}\n'.format(n)
        output_line += 'Sentiment:\tF1 Score:{}\tF1 Score SEval:{}\tF1 Score Neg:{}\tF1 Score Neut:{}\tF1 Score Pos:{}\n'.format(
            np.mean(f1_score_senti),
            0.5*(f1_score_senti[0] + f1_score_senti[2]),
            f1_score_senti[0],
            f1_score_senti[1],
            f1_score_senti[2]
        )

        ofile = open(ofname, 'wt')
        for id, pred in zip(ids, y_pred_senti_cls.tolist()):
            oline = '{}\t{}\n'.format(id, pred)
            ofile.write(oline)

    return output_line
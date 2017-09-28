from keras.utils.np_utils import probas_to_classes
from sklearn.metrics import f1_score
import numpy as np

def evaluate(model, test_iteraotr, experiment_name, ofname):
    inputs = test_iteraotr.input_data
    outputs = test_iteraotr.output_data
    names = test_iteraotr.names

    for i, o, n in zip(inputs, outputs, names):
        ofile = open(ofname, 'wt')
        y_test_senti = probas_to_classes(o)

        model_output = model.predict(i)
        sentence_embeddings = model_output[1]
        y_pred = model_output[0]

        y_pred_senti = y_pred

        y_pred_senti_cls = probas_to_classes(y_pred_senti)

        f1_score_senti = f1_score(y_test_senti, y_pred_senti_cls, average=None, pos_label=None, labels=[0, 1, 2])

        output_line = '{}\n'.format(n)
        output_line += 'Sentiment:\tF1 Score:{}\tF1 Score SEval:{}\tF1 Score Neg:{}\tF1 Score Neut:{}\tF1 Score Pos:{}\n'.format(
            np.mean(f1_score_senti),
            0.5 * (f1_score_senti[0] + f1_score_senti[2]),
            f1_score_senti[0],
            f1_score_senti[1],
            f1_score_senti[2]
        )

        for list in sentence_embeddings:
            slist = [str(x) for x in list.tolist()]

            file_output_line = '{}\n'.format(' '.join(slist))
            ofile.write(file_output_line)

    return output_line
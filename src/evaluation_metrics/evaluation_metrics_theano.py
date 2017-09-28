import numpy as np
import keras.backend as K
from theano import scan
from scipy.stats import pearsonr

def average_precision(y_true, y_pred):
    y_pred_sort_idx = K.T.argsort(y_pred, axis=0)[::-1]
    y_true_sorted = y_true[y_pred_sort_idx]

    true_cumsum = K.T.cumsum(y_true_sorted)
    true_range = K.T.arange(1, y_true.shape[0] + 1)
    true_sum = K.sum(y_true)

    #if no prediction is relevant just return 0
    return K.T.switch(K.T.eq(true_sum, 0), 0, K.sum((true_cumsum / true_range) * y_true_sorted) * (1 / true_sum))


def mean_average_precision(y_true, y_pred):
    y_pred = K.reshape(y_pred[:, 1], shape=(y_pred.shape[0] // 10, 10))
    y_true = K.reshape(y_true[:, 1], shape=(y_true.shape[0] // 10, 10))

    results, updates = scan(
        fn=average_precision,
        outputs_info=None,
        sequences=[y_true, y_pred]
        )

    return K.mean(results)

def pearsons_correlation(y_true, y_pred):
    y_true_mean = K.mean(y_true)
    y_pred_mean = K.mean(y_pred)

    true_diff_mean = y_true - y_true_mean
    pred_diff_mean = y_pred - y_pred_mean

    sum_prod_diff_mean = K.sum(true_diff_mean*pred_diff_mean)
    true_diff_mean_sq = K.sum(K.square(true_diff_mean))
    pred_diff_mean_sq = K.sum(K.square(pred_diff_mean))

    r = sum_prod_diff_mean/(K.sqrt(true_diff_mean_sq)*K.sqrt(pred_diff_mean_sq) + K.epsilon())

    return r

def f1_score_keras(y_true, y_pred):
    #convert probas to 0,1
    y_ppred = K.zeros_like(y_true)
    y_pred_ones = K.T.set_subtensor(y_ppred[K.T.arange(y_true.shape[0]), K.argmax(y_pred, axis=-1)], 1)

    #where y_ture=1 and y_pred=1 -> true positive
    y_true_pred = K.sum(y_true*y_pred_ones, axis=0)

    #for each class: how many where classified as said class
    pred_cnt = K.sum(y_pred_ones, axis=0)

    #for each class: how many are true members of said class
    gold_cnt = K.sum(y_true, axis=0)

    #precision for each class
    precision = K.T.switch(K.T.eq(pred_cnt, 0), 0, y_true_pred/pred_cnt)

    #recall for each class
    recall = K.T.switch(K.T.eq(gold_cnt, 0), 0, y_true_pred/gold_cnt)

    #f1 for each class
    f1_class = K.T.switch(K.T.eq(precision + recall, 0), 0, 2*(precision*recall)/(precision+recall))

    #return average f1 score over all classes
    return K.mean(f1_class)


def f1_score_taskB(y_true, y_pred):
    #convert probas to 0,1
    y_pred_ones = K.zeros_like(y_true)
    y_pred_ones[:, K.argmax(y_pred, axis=-1)] = 1

    #where y_ture=1 and y_pred=1 -> true positive
    y_true_pred = K.sum(y_true*y_pred_ones, axis=0)

    #for each class: how many where classified as said class
    pred_cnt = K.sum(y_pred_ones, axis=0)

    #for each class: how many are true members of said class
    gold_cnt = K.sum(y_true, axis=0)

    #precision for each class
    precision = K.switch(K.equal(pred_cnt, 0), 0, y_true_pred/pred_cnt)

    #recall for each class
    recall = K.switch(K.equal(gold_cnt, 0), 0, y_true_pred/gold_cnt)

    #f1 for each class
    f1_class = K.switch(K.equal(precision + recall, 0), 0, 2*(precision*recall)/(precision+recall))

    #return average f1 score over all classes
    return f1_class

def f1_score_semeval(y_true, y_pred):
    # convert probas to 0,1
    y_ppred = K.zeros_like(y_true)
    y_pred_ones = K.T.set_subtensor(y_ppred[K.T.arange(y_true.shape[0]), K.argmax(y_pred, axis=-1)], 1)

    # where y_ture=1 and y_pred=1 -> true positive
    y_true_pred = K.sum(y_true * y_pred_ones, axis=0)

    # for each class: how many where classified as said class
    pred_cnt = K.sum(y_pred_ones, axis=0)

    # for each class: how many are true members of said class
    gold_cnt = K.sum(y_true, axis=0)

    # precision for each class
    precision = K.T.switch(K.T.eq(pred_cnt, 0), 0, y_true_pred / pred_cnt)

    # recall for each class
    recall = K.T.switch(K.T.eq(gold_cnt, 0), 0, y_true_pred / gold_cnt)

    # f1 for each class
    f1_class = K.T.switch(K.T.eq(precision + recall, 0), 0, 2 * (precision * recall) / (precision + recall))

    #return average f1 score over all classes
    return (f1_class[0] + f1_class[2])/2.0

def precision_keras(y_true, y_pred):
    #convert probas to 0,1
    y_pred_ones = K.zeros_like(y_true)
    y_pred_ones[:, K.argmax(y_pred, axis=-1)] = 1

    #where y_ture=1 and y_pred=1 -> true positive
    y_true_pred = K.sum(y_true*y_pred_ones, axis=0)

    #for each class: how many where classified as said class
    pred_cnt = K.sum(y_pred_ones, axis=0)

    #precision for each class
    precision = K.switch(K.equal(pred_cnt, 0), 0, y_true_pred/pred_cnt)
    
    #return average f1 score over all classes
    return K.mean(precision)


def f1_score_task3(y_true, y_pred):
    #convert probas to 0,1
    y_ppred = K.zeros_like(y_true)
    y_pred_ones = K.T.set_subtensor(y_ppred[K.T.arange(y_true.shape[0]), K.argmax(y_pred, axis=-1)], 1)

    #where y_ture=1 and y_pred=1 -> true positive
    y_true_pred = K.sum(y_true*y_pred_ones, axis=0)

    #for each class: how many where classified as said class
    pred_cnt = K.sum(y_pred_ones, axis=0)

    #for each class: how many are true members of said class
    gold_cnt = K.sum(y_true, axis=0)

    #precision for each class
    precision = K.T.switch(K.T.eq(pred_cnt, 0), 0, y_true_pred/pred_cnt)

    #recall for each class
    recall = K.T.switch(K.T.eq(gold_cnt, 0), 0, y_true_pred/gold_cnt)

    #f1 for each class
    f1_class = K.T.switch(K.T.eq(precision + recall, 0), 0, 2*(precision*recall)/(precision+recall))

    #return average f1 score over all classes
    return f1_class[1]

if __name__ == '__main__':
    y_true = np.array([0.0, 0.2, 0.4, 0.4])
    y_pred = np.array([0.0, 0.2, 0.4, 0.4])

    print(pearsonr(y_true, y_pred))

    y_true = K.variable(value=y_true)
    y_pred = K.variable(value=y_pred)
    print(K.eval(pearsons_correlation(y_true, y_pred)))

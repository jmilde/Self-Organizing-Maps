import tensorflow as tf
import numpy as np
from collections import Counter

def normalize(raw_data, normalize_by_column=False):
    # if True, assume all data is on common scale
    # if False, normalise to [0 1] range along each column
    if normalize_by_column:
        col_maxes = raw_data.max(axis=0)
        data = raw_data / col_maxes[np.newaxis, :]
    else:
        data = raw_data / raw_data.max()
    return data

def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`.
    if tensor `x` is given, converts and uses it as default.
    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def pipe(data, prefetch=1, repeat=-1, name='pipe', **kwargs):
    """
    data: np.array
    see `tf.data.Dataset.from_generator`."""
    with tf.variable_scope(name):
        return tf.data.Dataset.from_tensor_slices(data) \
                              .repeat(repeat) \
                              .prefetch(prefetch) \
                              .make_one_shot_iterator() \
                              .get_next()
def profile(sess, wrtr, run, feed_dict= None, prerun= 5, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wrtr.add_run_metadata(meta, tag)


def analyze(labels, code_lbl, threshold, counts, max_acc=1):
    """
          labels: cluster map
        code_lbl: list of 0/1 depending on if the Code appears or not
        treshold: 0.xx how many % of a cluster have to be of the chosen code for it to become relevant
          counts: how many instances must be in a cluster
         max_acc: only clusters with an accuracy smaller than max_acc are selected

    return: pred_counts = number of instances per cluster
             pred_acc    = % of instances in the cluster being of the code
             best_pred   = cluster above treshold and count
    """
    n_clusters = len(set(labels))

    pred_match = np.zeros(n_clusters)
    for idx, label in enumerate(labels):
        if code_lbl[idx] == 1:
            pred_match[label] += 1

    # array that counts how many instances are in the cluster
    pred_counts = np.array([x[1] for x in list(Counter(labels).items())])

    pred_acc = np.array([(b/a) if b!=0 else 0 for a, b in zip(pred_counts, pred_match)])

    best_pred = [[idx, x, pred_counts[idx]] for idx, x in enumerate(pred_acc)
                 if x>threshold and pred_counts[idx]>counts and x<max_acc]

    return pred_counts, pred_acc, best_pred

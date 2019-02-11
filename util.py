import tensorflow as tf
import numpy as np

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

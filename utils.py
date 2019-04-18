import shutil
import numpy as np
import tensorflow as tf


def silent_remove(d):
    try:
        shutil.rmtree(d)
    except FileNotFoundError:
        pass


class Logger:
    """Logging in Tensorboard without Tensorflow ops (or TensorboardX which does not work
    in this conda environment for some reason).
    """
    def __init__(self, logdir):
        silent_remove(logdir)
        self.writer = tf.summary.FileWriter(logdir)

    def log_scalar(self, tag, value, step):
        """Log scalar value.
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Log histogram of numpy array of values.
        """
        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)

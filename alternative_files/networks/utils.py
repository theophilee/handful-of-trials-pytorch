import tensorflow as tf
import torch

# TODO use truncnorm from scipy.stats instead of tensorflow, and get rid of tensorflow everywhere
def truncated_normal(size, std):
    # This initialization does not exist in cartpole and is important for rapid early training in cartpole.
    # Sample from a Gaussian and reject samples more than two standard deviations away from the mean.
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)
    val = sess.run(tf.truncated_normal(shape=size, stddev=std))
    sess.close()
    return torch.tensor(val, dtype=torch.float32)
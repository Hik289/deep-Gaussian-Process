import sys
sys.path.append(".")

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

from dataset import DataSet
import utils
import likelihoods
from DGP import DGP
import tensorflow as tf
import numpy as np
import losses

def import_dataset(dataset):

    train_X = np.loadtxt('data/' + 'Xtrain', delimiter=' ')
    train_Y = np.loadtxt('data/' + 'ytrain', delimiter=' ')
    train_Y = np.reshape(train_Y, (-1, 1))
    test_X = np.loadtxt('data/' + 'Xtest', delimiter=' ')
    test_Y = np.loadtxt('data/' + 'ytest', delimiter=' ')
    test_Y = np.reshape(test_Y, (-1, 1))

    data = DataSet(train_X, train_Y)
    test = DataSet(test_X, test_Y)

    return data, test


if __name__ == '__main__':
    FLAGS = utils.get_flags()

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test = import_dataset(FLAGS.dataset)

    error_rate = losses.RootMeanSqError(data.Dout)

    like = likelihoods.Gaussian()

    optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    dgp = DGP(like, data.num_examples, data.X.shape[1], data.Y.shape[1], FLAGS.nl, FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree, FLAGS.is_ard, FLAGS.local_reparam, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, FLAGS.learn_Omega)

    dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
                 FLAGS.display_step, test, FLAGS.mc_test, error_rate, FLAGS.duration, FLAGS.less_prints)

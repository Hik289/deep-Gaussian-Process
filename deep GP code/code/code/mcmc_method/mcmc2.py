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

from mcmc import MCMC

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def h(x):
    return np.exp(-np.square(x)) * (2.0 * x)

def f(x):
    return h(h(x))

def generate_toy_data():

    N = 50
    DATA_X = np.random.uniform(-5.0, 5.0, [N, 1])

    true_log_lambda = -2.0
    true_std = np.exp(true_log_lambda) / 2.0  # 0.1
    DATA_y = f(DATA_X) + np.random.normal(0.0, true_std, [N, 1])

    Xtest = np.asarray(np.arange(-10.0, 10.0, 0.1))
    Xtest = Xtest[:, np.newaxis]
    ytest = f(Xtest)

    data = DataSet(DATA_X, DATA_y)
    test = DataSet(Xtest, ytest, shuffle=False)

    return data, test


if __name__ == '__main__':
    FLAGS = utils.get_flags()

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    data, test = generate_toy_data()

    error_rate = losses.RootMeanSqError(data.Dout)

    like = likelihoods.Gaussian()

    optimizer = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate)

    dgp = DGP(like, data.num_examples, data.X.shape[1], data.Y.shape[1], FLAGS.nl, FLAGS.n_rff, FLAGS.df, FLAGS.kernel_type, FLAGS.kernel_arccosine_degree, FLAGS.is_ard, FLAGS.feed_forward, FLAGS.q_Omega_fixed, FLAGS.theta_fixed, FLAGS.learn_Omega, FLAGS.seed)


    dgp.learn(data, FLAGS.learning_rate, FLAGS.mc_train, FLAGS.batch_size, FLAGS.n_iterations, optimizer,
                 FLAGS.display_step, test, FLAGS.mc_test, error_rate)

    layers = dgp.session.run(dgp.layer, feed_dict={dgp.X:test.X, dgp.Y:test.Y, dgp.mc:FLAGS.mc_test})
    predictions_variational_F1 = np.zeros([test.Y.shape[0],FLAGS.mc_test])
    predictions_variational_F2 = np.zeros([test.Y.shape[0],FLAGS.mc_test])
    for i in range(FLAGS.mc_test):
        predictions_variational_F1[:,i] = layers[1][i,:,0]
        predictions_variational_F2[:,i] = layers[2][i,:,0]


    samples_F1, samples_F2, predictions_F1, predictions_F2 = MCMC(data.X, data.Y, test.X)




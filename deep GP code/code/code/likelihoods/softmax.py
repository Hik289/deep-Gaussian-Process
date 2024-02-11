import tensorflow as tf
from . import likelihood
import utils

class Softmax(likelihood.Likelihood):

    def log_cond_prob(self, output, latent_val):
        return tf.reduce_sum(output * latent_val, 2) - utils.logsumexp(latent_val, 2)

    def predict(self, latent_val):

        logprob = latent_val - tf.expand_dims(utils.logsumexp(latent_val, 2), 2)
        return tf.exp(logprob)

    def get_params(self):
        return None

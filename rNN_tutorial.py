'''
    rNN_tutorial.py

    This is an implementation of the Tensorflow rNN word2vec tutorial found at https://www.tensorflow.org/tutorials/

    Some of this code may be highly influenced or taken directly from the tutorial and not my own.

    Word2vec Tutorial
    https://www.tensorflow.org/tutorials/word2vec

    Why word2vec?
    Model relationships / similarity between words
    Reduce sparcity of one-hot encoded vector representations of dictionaries
    Efficient predictive model for learning word embeddings from text

    Vector Space Models (VSM)
    Mapping words to continuous vector space where similar words are nearby (relies on Distrbutional Hypothesis)

    Primary methods VSM
    Latent Semantic Analysis - Counts of words in sample using SVD (latent vectors - mapping to lower dimensional continuous space), then using cosine similarity to asses similarity
    Predictive Methods - Neural Network Probabilistic models

    Word2vec algorithms

    Continuous Bag-of-Words CBOW - predicts target word from context (context is one observation) - smaller datasets
    Skip-Gram - predicts context from target words (context-target pair as one observation) - larger datasets

    Can use maximum-likelihood for gradient based optimisation naively.

    Negative Sampling
    Maximise p(word in context in dataset) + k E[samples from noise distribution].
    Maximised when high probabilities are assigned to real words and low Probabilistic to noise words.
    Approximates softmax MLE in the limits while increasing with size of noise words instead of vocabulary.

    Target - word for which testing and prediction will apply
    Context - A window of n words on either side of the target word - implementations can vary


'''

import tensorflow as tf
import numpy as np
import math
from word2vec_basic.py import maybe_download
from word2vec_basic.py import read_data
from word2vec_basic.py import build_dataset

# Word2vec - Skip-Gram Visualisation

# Define embeding matrix
embedings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# Noise contrastive estimation loss defined as logistic regression model:
#   W: vocabulary_size x embedding_size     B: vocabulary_size x 1
W_nce = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
B_nce = tf.Variable(tf.zeros([vocabulary_size]))

# Input placeholder
X = tf.placeholder(tf.int32, shape=[batch_size])
Y_ = tf.placeholder(tf.int32, shape=[batch_size, 1])

# embedings for each source word in batch
X_emb = tf.nn.embeding_lookup(embedings, X)

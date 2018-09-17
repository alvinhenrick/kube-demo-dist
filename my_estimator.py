import string

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence

tf.logging.set_verbosity(tf.logging.INFO)

vocab_size = 5000
sentence_size = 200
embedding_size = 50

(x_train_variable, y_train), (x_test_variable, y_test) = imdb.load_data(num_words=vocab_size)
x_train = sequence.pad_sequences(
    x_train_variable,
    maxlen=sentence_size,
    padding='post',
    value=0)
x_test = sequence.pad_sequences(
    x_test_variable,
    maxlen=sentence_size,
    padding='post',
    value=0)

x_len_train = np.array([min(len(x), sentence_size) for x in x_train_variable])
x_len_test = np.array([min(len(x), sentence_size) for x in x_test_variable])

word_index = imdb.get_word_index()
index_offset = 3


# Create an input functions reading a file using the Dataset API
# Then provide the results to the Estimator API
def parser(x, length, y):
    features = {"x": x, "len": length}
    return features, y


def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train_variable))
    dataset = dataset.batch(100)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, x_len_test, y_test))
    dataset = dataset.batch(100)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


head = tf.contrib.estimator.binary_classification_head()


def cnn_model_fn(features, labels, mode, params):
    input_layer = tf.contrib.layers.embed_sequence(
        features['x'], vocab_size=vocab_size, embed_dim=embedding_size,
        initializer=params['embedding_initializer'])

    training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_emb = tf.layers.dropout(inputs=input_layer,
                                    rate=0.2,
                                    training=training)

    conv = tf.layers.conv1d(
        inputs=dropout_emb,
        filters=32,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    # Global Max Pooling
    pool = tf.reduce_max(input_tensor=conv, axis=1)

    hidden = tf.layers.dense(inputs=pool, units=250, activation=tf.nn.relu)

    dropout_hidden = tf.layers.dropout(inputs=hidden,
                                       rate=0.2,
                                       training=training)

    logits = tf.layers.dense(inputs=dropout_hidden, units=1)

    # This will be None when predicting
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    optimizer = tf.train.AdamOptimizer()

    def _train_op_fn(loss):
        return optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

    return head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        train_op_fn=_train_op_fn)


def text_to_index(sentence):
    # Remove punctuation characters except for the apostrophe
    translator = str.maketrans('', '', string.punctuation.replace("'", ''))
    tokens = sentence.translate(translator).lower().split()
    return np.array([1] + [word_index[t] + index_offset if t in word_index else 2 for t in tokens])

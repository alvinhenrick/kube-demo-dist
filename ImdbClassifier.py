import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence

from my_estimator import cnn_model_fn, text_to_index, sentence_size


class ImdbClassifier(object):

    def __init__(self):
        model_dir = os.getenv('MODEL_DIR', './imdb_model')
        params = {'embedding_initializer': tf.random_uniform_initializer(-1.0, 1.0)}

        # Create a custom estimator using my_model_fn to define the model
        tf.logging.info("Before classifier construction")
        estimator = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=model_dir,
            params=params)  # Path to where checkpoints etc are stored
        self.estimator = estimator
        tf.logging.info("...done constructing classifier")

    def predict(self, input_data, feature_names):
        indexes = [text_to_index(sentence) for sentence in input_data]
        x = sequence.pad_sequences(indexes,
                                   maxlen=sentence_size,
                                   truncating='post',
                                   padding='post',
                                   value=0)
        length = np.array([min(len(x), sentence_size) for x in indexes])
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x, "len": length}, shuffle=False)
        predict_results = self.estimator.predict(input_fn=predict_input_fn)

        return [[x["class_ids"][0]] for x in predict_results]

# if __name__ == '__main__':
#     t = ImdbClassifier()
#     results = t.predict([
#         'I really liked the movie!', 'Hated every second of it...'])
#     print(results)

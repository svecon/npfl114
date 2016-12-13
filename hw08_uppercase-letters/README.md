# HW-08 uppercase_letters (4pts, due Dec 05)

Implement network, which is given an English sentence in lowercase letters and tries to uppercase appropriate letters. Use the labs06/en-ud-train.txt as training data, labs06/en-ud-dev.txt as development data and labs06/en-ud-test.txt as testing data.

Start with the labs06/uppercase-letters-skeleton.py file, which loads the data, remaps characters to integers, generates random batches and saves summaries.

Represent letters either as one-hot vectors (tf.one_hot) or using trainable embeddings (tf.nn.embedding_lookup), and use bidirectional LSTM/GRU (using tf.nn.bidirectional_dynamic_rnn) combined with a linear classification layer with softmax. Report test set accuracy. For your information, straightforward approach with small hyperparameter search on development data has test accuracy of 97.63%.

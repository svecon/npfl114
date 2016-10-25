# HW-03 mnist_dropout (2pts, due Nov 07)

Using the MNIST example from labs03/1-mnist.py, implement dropout (using tf.nn.dropout). During training, allow specifying dropout probability for the input layer and for the hidden layer separately. Then perform hyperparameter search using:
* input layer dropout keep probability (0.5,0.7,0.8,0.9,1)
* hidden layer dropout keep probability (0.3,0.4,0.5,0.6,1)

and report both development set accuracy for all hyperparameters and test set accuracy for the best hyperparameters.
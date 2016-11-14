# HW-05 mnist_conv (3-5pts, due Nov 14)

Try achieving as high accuracy on the MNIST test set as possible (you can start from labs03/1-mnist.py, byt you can modify it freely). Nevertheless, remember that you should not perform hyperparameter search on the test set (when you design network architecture, you should perform hyperparameter search on the development set, and measure the test set accuracy only with the best hyperparameters; and optionally repeat with modified architecture). You will be awarded points according to the accuracy achieved:
* 99.1 test set accuracy: 3 points
* 99.25 test set accuracy: 4 points
* 99.4 test set accuracy: 5 points

You should use convolution (see tf.contrib.layers.convolution2d or directly tf.nn.conv2d).

To solve this task, send me a source code I can execute (using python source.py) which trains a neural network and prints the test set accuracy on standard output (in less than a day :-).

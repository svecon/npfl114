# HW-01 mnist_layers_activations (3pts, due Oct 31)

Modify one of the MNIST examples from labs03 so that it uses the following hyperparameters:
* layers: number of hidden layers (1-3)
* activation: activation function, either tf.tanh or tf.nn.relu

Then implement hyperparameter search – find the values of hyperpamaters resulting in the best accuracy on the development set (mnist.validation) and using these hyperparameters compute the accuracy on the test set (mnist.test).

# HW-02 mnist_training (2pts, due Nov 07)

Using the MNIST example labs03/1-mnist.py, try the following optimizers:
* standard SGD (tf.train.GradientDescentOptimizer), with batch sizes (1,10,50) and learning rates (0.01,0.001,0.0001)
* SGD with exponential learning rate decay (use tf.train.exponential_decay), with batch sizes (1,10,50) and the following (starting learning rate, final learning rate) pairs: (0.01,0.001), (0.01,0.0001), (0.001, 0.0001)
* SGD with momentum (tf.train.MomentumOptimizer), with batch sizes (1,10,50), learning rates (0.01,0.001,0.0001) and momentum 0.9
* Adam optimizer (tf.train.AdamOptimizer), with batch sizes (1,10,50) and learning rates (0.002,0.001,0.0005)

Report the development set accuracy for all the listed possibilities.
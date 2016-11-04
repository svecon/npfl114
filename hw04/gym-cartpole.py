from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import os

import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

class Network:
    OBSERVATIONS = 4
    HIDDEN_SIZE = 10
    LABELS = 2

    def __init__(self, threads=1, logdir=None, expname=None, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        if logdir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.summary_writer = tf.train.SummaryWriter(("{}/{}-{}" if expname else "{}/{}").format(logdir, timestamp, expname), flush_secs=10)
        else:
            self.summary_writer = None

    def construct(self):
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
                self.labels = tf.placeholder(tf.int64, [None], name="labels")

            hidden_layer1 = tf_layers.fully_connected(self.observations, num_outputs=self.HIDDEN_SIZE, activation_fn=tf.nn.relu, scope="hidden_layer_1")
            hidden_layer2 = tf_layers.fully_connected(hidden_layer1, num_outputs=self.HIDDEN_SIZE, activation_fn=tf.nn.relu, scope="hidden_layer_2")
            output_layer = tf_layers.fully_connected(hidden_layer2, num_outputs=self.LABELS, activation_fn=None, scope="output_layer")

            self.action = tf.argmax(output_layer, 1)

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.labels, scope="loss")

            # Global step
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)
            self.accuracy = tf_metrics.accuracy(self.action, self.labels)

            # Summaries
            self.summaries = {"training": tf.merge_summary([tf.scalar_summary("train/loss", loss),
                                                            tf.scalar_summary("train/accuracy", self.accuracy)])}

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/action", self.action)
            self.saver = tf.train.Saver(max_to_keep=None)

            # Initialize the variables
            self.session.run(tf.initialize_all_variables())

        # Finalize graph and log it if requested
        self.session.graph.finalize()
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, observations, labels, summaries=False, run_metadata=False):
        if (summaries or run_metadata) and not self.summary_writer:
            raise ValueError("Logdir is required for summaries or run_metadata.")

        args = {"feed_dict": {self.observations: observations, self.labels: labels}}
        targets = [self.training]
        if summaries:
            targets.append(self.summaries["training"])
        if run_metadata:
            args["options"] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            args["run_metadata"] = tf.RunMetadata()

        results = self.session.run(targets, **args)
        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step - 1)
        if run_metadata:
            self.summary_writer.add_run_metadata(args["run_metadata"], "step{:05}".format(self.training_step - 1))


    # Save the graph
    def save(self, directory, path):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.session, os.path.join(directory,path))


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="1-gym-save", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--epochs", default=100000, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    args = parser.parse_args()

    # Construct the network
    network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
    network.construct()

    # TODO: Train the network
    from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
    from numpy import genfromtxt
    
    data = genfromtxt("../labs04/gym-cartpole-data.txt", delimiter=' ')
    traindata = DataSet(data[:,:-1], data[:,-1], reshape=False)

    # Train
    for i in range(args.epochs):
        while traindata.epochs_completed == i:
            observations, labels = traindata.next_batch(args.batch_size)
            network.train(observations, labels, network.training_step % 100 == 0, network.training_step == 0)

    # Save the network
    network.save("networks", "1-gym-2hidden")

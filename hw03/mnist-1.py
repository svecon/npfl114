from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

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

    def construct(self, hidden_layer_size):
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
                self.labels = tf.placeholder(tf.int64, [None], name="labels")

            flattened_images = tf_layers.flatten(self.images, scope="preprocessing")

            self.input_layer_dropout_rate = tf.placeholder(tf.float32, [])
            input_layer_dropout = tf.nn.dropout(flattened_images, self.input_layer_dropout_rate)
            
            hidden_layer = tf_layers.fully_connected(input_layer_dropout, num_outputs=hidden_layer_size, activation_fn=tf.nn.relu, scope="hidden_layer")
            
            self.hidden_layer_dropout_rate = tf.placeholder(tf.float32, [])
            hidden_layer_dropout = tf.nn.dropout(hidden_layer, self.hidden_layer_dropout_rate)
            
            output_layer = tf_layers.fully_connected(hidden_layer_dropout, num_outputs=self.LABELS, activation_fn=None, scope="output_layer")
            self.predictions = tf.argmax(output_layer, 1)

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.labels, scope="loss")
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)
            self.accuracy = tf_metrics.accuracy(self.predictions, self.labels)

            # Summaries
            self.summaries = {"training": tf.merge_summary([tf.scalar_summary("train/loss", loss),
                                                            tf.scalar_summary("train/accuracy", self.accuracy)])}
            for dataset in ["dev", "test"]:
                self.summaries[dataset] = tf.scalar_summary(dataset+"/accuracy", self.accuracy)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

        # Finalize graph and log it if requested
        self.session.graph.finalize()
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, images, labels, input_dropout_prob, hidden_dropout_prob, summaries=False, run_metadata=False):
        if (summaries or run_metadata) and not self.summary_writer:
            raise ValueError("Logdir is required for summaries or run_metadata.")

        args = {"feed_dict": {
                self.images: images,
                self.labels: labels,
                self.input_layer_dropout_rate: input_dropout_prob,
                self.hidden_layer_dropout_rate: hidden_dropout_prob,
        }}
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

    def evaluate(self, dataset, images, labels, summaries=False):
        if summaries and not self.summary_writer:
            raise ValueError("Logdir is required for summaries.")

        targets = [self.accuracy]
        if summaries:
            targets.append(self.summaries[dataset])

        results = self.session.run(targets, {self.images: images, self.labels: labels, self.input_layer_dropout_rate: 1, self.hidden_layer_dropout_rate: 1})
        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step)
        return results[0]


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="4-mnist-using-contrib", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    from tensorflow.examples.tutorials.mnist import input_data
    
    for input_dropout_prob in (0.8, 0.9, 1):
        for hidden_dropout_prob in (0.8, 0.9, 1):

            mnist = input_data.read_data_sets("mnist_data/", reshape=False)

            # Construct the network
            network = Network(threads=args.threads, logdir=args.logdir, expname='{}_idp={}_hdp={}'.format(args.exp, input_dropout_prob, hidden_dropout_prob))
            network.construct(100)

            # Train
            for i in range(args.epochs):
                while mnist.train.epochs_completed == i:
                    images, labels = mnist.train.next_batch(args.batch_size)
                    network.train(images, labels, input_dropout_prob, hidden_dropout_prob, network.training_step % 100 == 0, network.training_step == 0)

                network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)
                network.evaluate("test", mnist.test.images, mnist.test.labels, True)

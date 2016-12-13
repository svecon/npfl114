# The `resnet_v1_50.ckpt` can be downloaded from http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz

from __future__ import division
from __future__ import print_function

import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
import tensorflow.contrib.slim.nets
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

import imagenet_classes
import subcaltech_classes

class Network:
    WIDTH = 224
    HEIGHT = 224
    CLASSES = 50

    def __init__(self, checkpoint, threads, logdir=None):
        # Create the session
        self.session = tf.Session(graph = tf.Graph(), config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                            intra_op_parallelism_threads=threads))

        if logdir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.summary_writer = tf.train.SummaryWriter("{}/{}".format(logdir, timestamp), flush_secs=10)
        else:
            self.summary_writer = None

        with self.session.graph.as_default():
            # Construct the model
            with tf.name_scope("inputs"):
                self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 3], name="images")
                self.labels = tf.placeholder(tf.int64, [None], name="labels")

            with tf_slim.arg_scope(tf_slim.nets.resnet_v1.resnet_arg_scope(is_training=False)):
                resnet, _ = tf_slim.nets.resnet_v1.resnet_v1_50(self.images, num_classes=None)
            self.resnetoutput = tf.squeeze(resnet, [1, 2])
            
            # Load the checkpoint
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, checkpoint)

            learnt_variables = set(tf.all_variables())

            # Add more layers
            output_layer = tf_layers.fully_connected(self.resnetoutput, num_outputs=self.CLASSES, activation_fn=None, scope="output_layer")
            self.predictions = tf.argmax(output_layer, 1)

            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "output_layer")

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.labels, scope="loss")
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.training = tf.train.AdamOptimizer(name='adam').minimize(loss, global_step=self.global_step, var_list=train_vars)
            self.accuracy = tf_metrics.accuracy(self.predictions, self.labels)

            # Summaries
            self.summaries = {"training": tf.merge_summary([tf.scalar_summary("train/loss", loss),
                                                            tf.scalar_summary("train/accuracy", self.accuracy)])}
            for dataset in ["dev", "test"]:
                self.summaries[dataset] = tf.scalar_summary(dataset+"/accuracy", self.accuracy)

            self.session.run(tf.initialize_variables( set(tf.all_variables())-learnt_variables ))

            # JPG loading
            self.jpeg_file = tf.placeholder(tf.string, [])
            self.jpeg_data = tf.image.resize_image_with_crop_or_pad(tf.image.decode_jpeg(tf.read_file(self.jpeg_file), channels=3), self.HEIGHT, self.WIDTH)

        # Finalize graph and log it if requested
        self.session.graph.finalize()
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    def load_jpeg(self, jpeg_file):
        return self.session.run(self.jpeg_data, {self.jpeg_file: jpeg_file})

    def predict(self, image):
        return self.session.run(self.predictions, {self.images: [image]})[0]

    # Save the graph
    def save(self, directory, path):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.session, os.path.join(directory,path))

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, images, labels, summaries=False, run_metadata=False):
        if (summaries or run_metadata) and not self.summary_writer:
            raise ValueError("Logdir is required for summaries or run_metadata.")

        args = {"feed_dict": {self.images: images, self.labels: labels}}
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

        results = self.session.run(targets, {self.images: images, self.labels: labels})

        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step)
        return results[0]


if __name__ == "__main__":
    # Parse arguments
    import argparse
    import pandas as pd
    from DataSet import DataSet
    # from StringIO import StringIO
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("images", type=str, nargs='+', help="Image files.")
    parser.add_argument("--checkpoint", default="resnet_v1_50.ckpt", type=str, help="Name of ResNet50 checkpoint.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
    args = parser.parse_args()

    # Load the network
    network = Network(args.checkpoint, args.threads, logdir=args.logdir)

    data = pd.read_csv('train.txt', delimiter=" ", header=None).values
    traindata = DataSet(
        np.array(list(map(network.load_jpeg, data[:,0]))),
        np.array(list(map(subcaltech_classes.subcaltech_classes.index, data[:,-1]))),
        reshape=False)

    data = pd.read_csv('test.txt', delimiter=" ", header=None).values
    testdata = DataSet(
        np.array(list(map(network.load_jpeg, data[:,0]))),
        np.array(list(map(subcaltech_classes.subcaltech_classes.index, data[:,-1]))),
        reshape=False)

    # # Train
    for i in range(args.epochs):
        while traindata.epochs_completed == i:
            images, labels = traindata.next_batch(args.batch_size)
            print('running next batch')
            print(labels)
            network.train(images, labels, network.training_step % 100 == 0, network.training_step == 0)

        accuracy = network.evaluate("dev", testdata.images, testdata.labels, True)
        print('Accuracy: {}', accuracy)

    # # Save the network
    # network.save("networks", "subcaltech_nn")


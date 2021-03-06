# HW-06 resnet_subcaltech (5pts, due Nov 28)

Implement network which will perform image classification on Sub-Caltech50 dataset (this dataset was created for this task as a subset of Caltech101). The dataset contains images classified in 50 classes and has explicit train/test partitioning (it does not have explicit development partition, use some amount of training data if you need one).

In order to implement the image classification, use pre-trained ResNet50 network to extract image features (we do not use ResNet101 nor ResNet152 as they are more computationally demanding). To see how ResNet50 can be used to classify an image on the ImageNet classes, see the labs05/resnet50.py. When using the ResNet50 to extract features, pass num_classes=None when creating the network, and the network will return 2048 image features instead of logits of 1000 classes.

The goal of this task is to train an image classifier using the image features precomputed by ResNet50, and report the testing accuracy. The best course of action is probably to precompute the image features once (for both training and testing set) and save them to disc, and then train the classifier using the precomputed features. As for the classifier model, it is probably enough to create a fully connected layer to 50 neurons with softmax (without ReLU).

Bonus: if you are interested, you can finetune the classifier including the ResNet50 and get additional points for it. After you train the classifier as described above, put both the ResNet50 and the pretrained classifier in one Graph, and continue training including the ResNet50 (you need to pass is_training=True during ResNet construction).

## Instructions

- Download dataset http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/subcaltech-50.zip
- Download ResNet50 model http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
- Run resnet50_extract-features.py to get 2048 features per image
- Run resnet50_last-layer.py to train last layer of the network
- Run resnet50_tune.py to further train the entire network

Instead of extracting the features and training the model separately, you can train the last layer using the full network by running resnet50_full-graph.py
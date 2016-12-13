# HW-07 sequence_generation (4pts, due Nov 28)

Implement network which performs sequence generation via LSTM/GRU. Note that for training purposes, we will be using very low-level approach.

The goal is to predict the labs06/international-airline-passengers.tsv sequence. Start with the labs06/sequence-generation-skeleton.py file, which loads the data and supports producing image summaries with the predicted sequence.

For training, construct an unrolled series of LSTM/GRU cells, using training portion of gold data as input, predicting the next value in the training sequence (the LSTM/GRU output contains several numbers, so use additional linear layer with one output, and MSE loss). In every epoch, train the same sequence several times (500 is the default in the script).

For prediction, use the last output state from the training portion of the network, and construct another unrolled series of LSTM/GRU cells, this time using the prediction from previous step as input.

Report results of both LSTM and GRU, each with 8, 10 and 12 cells (by sending the logs of the 6 runs).

#!/bin/bash
set -e
sizes="8 10 12"
types="LSTM GRU"
for size in $sizes; do
    for celltype in $types; do
        python sequence-generation.py --threads 8 --rnn_cell $celltype --rnn_cell_dim $size
    done
done

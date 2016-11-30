#!/bin/sh
set -e
for rnn in LSTM GRU
do
	for dim in 10 20 30 50
    do
    	for embedding in 0 100 150 200
        do
            python uppercase-letters.py --rnn_cell=$rnn --rnn_cell_dim=$dim --embedding=$embedding --threads=8
        done
    done
done

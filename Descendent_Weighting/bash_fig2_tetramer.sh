#!/bin/bash
for sim in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
	python3 DMC_rs_H2O_tetramer_fig2.py .1 1000000 10000 10000 tetramer/tetramer_sim"${sim}"_uniform/tetramer_sim"${sim}" >> output_log &

done
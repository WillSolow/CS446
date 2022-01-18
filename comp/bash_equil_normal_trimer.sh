#!/bin/bash
for prop in .5 .25 .05 .025 .005; do
	for test in 0 1 2 3 4;  do 
		python3 DMC_rs_H2O_trimer.py "${prop}" equil_normal r_trimer.npy trimer/sim"${test}" >> output_log &
	done
done
#!/bin/bash
for dt in .01 .05 .1 .5 1 2 3 4 5 7 9 10 15 20; do
	for test in 0 1 2 3 4;  do 
		python3 DMC_rs_H2O_tetramer_fig3.py "${dt}" 10000 tetramer/tetramer_dt"${dt}"_onemol/tetramer_dt_"${test}" >> output_log &
	done
done
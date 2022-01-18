#!/bin/bash
for wlk in 100 500 1000 2000 3000 4000 5000 7000 10000 20000 50000 100000; do
	for test in 0 1 2 3 4;  do 
		python3 DMC_rs_H2O_tetramer_fig1.py .1 1000000 "${wlk}" 10000 tetramer/tetramer_wlk"${wlk}"/tetramer_wlk"${wlk}"_"${test}" >> output_log &
	done
done
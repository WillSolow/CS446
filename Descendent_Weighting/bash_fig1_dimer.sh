#!/bin/bash
for wlk in 100 500 2000 3000 4000 7000 100000; do
	for test in 0 1 2 3 4;  do 
		python3 DMC_rs_H2O_dimer_fig1.py .1 1000000 "${wlk}" 10000 dimer/dimer_"${wlk}"k_"${test}" >> output_log &
	done
done
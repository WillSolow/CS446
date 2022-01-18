#!/bin/bash
for prop in 7 10 20; do
	for test in 0 1 2 3 4;  do 
		python3 DMC_rs_H2O_trimer.py "${prop}" random m_trimer.xyz trimer/sim"${test}" >> output_log &
		python3 DMC_rs_H2O_trimer.py "${prop}" equil_random r_trimer.npy trimer/sim"${test}" >> output_log &
	done
done
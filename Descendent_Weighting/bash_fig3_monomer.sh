#!/bin/bash
for dt in 3 4 7 9 15 20; do
	for test in 0 1 2 3 4;  do 
		python3 DMC_rs_H2O_monomer_fig3.py "${dt}" 10000 monomer/monomer_dt"${dt}"_"${test}" >> output_log &
	done
done
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

sequence=S1_Box
total=`ls -lah ../Data/HEVA_Validate/$sequence'_1_C1'/Image/*.png | wc -l`
total=$(( total-1 ))

echo $total
 
for ((idx=0; idx<=total; idx++))
do
	echo $idx
	python frame_fit.py $sequence $idx
	exit
done

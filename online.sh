#!/bin/bash

utils=(0.2 0.4 0.6)
algos=("edf+spt" "edf+bin")
gs_per_node=(1 2 4 8)
dvfs=off

for algo in "${algos[@]}"
do
  for gpn in "${gs_per_node[@]}"
  do
    for util in "${utils[@]}"
    do
      python main.py online-${util} ${algo}-${dvfs}-1.0 ${gpn} 
    done
  done
done


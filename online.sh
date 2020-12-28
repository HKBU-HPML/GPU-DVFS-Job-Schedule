#!/bin/bash

utils=(0.2 0.4 0.6 0.8 1.0)
algos=("edf+wf")
gs_per_node=(1 2 4 8)

for algo in "${algos[@]}"
do
  for gpn in "${gs_per_node[@]}"
  do
    for util in "${utils[@]}"
    do
      python main.py online-${util} ${algo}-off-1.0 ${gpn} 
    done
  done
done


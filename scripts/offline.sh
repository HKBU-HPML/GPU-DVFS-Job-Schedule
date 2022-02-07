#!/bin/bash

utils=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6)
algos=("edf+spt" "edf+bf" "edf+wf" "lpt+ff")
dvfs=off
gs_per_node=(16)

for gpn in "${gs_per_node[@]}"
do
  for algo in "${algos[@]}"
  do
    for util in "${utils[@]}"
    do
      echo ${algo} ${gpn} ${util} dvfs-${dvfs}
      python main.py offline-${util} ${algo}-${dvfs}-1.0 ${gpn} 1>/dev/null 2>&1
    done
  done
done


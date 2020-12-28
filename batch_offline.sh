#!/bin/bash

utils=(0.2 0.4 0.6 0.8 1.0)

for util in "${utils[@]}"
do
  python batch.py offline-${util} 
done


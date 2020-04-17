from __future__ import print_function
import os, glob, sys
from random import sample, randint
import json, yaml
from cluster import *
from job import sim_job
import numpy as np
import random
from job_master import *

job_set = 'offline_0.2'
adopted_algo = '1-edf+wf-on-1.0'
if len(sys.argv) == 2:
    adopted_algo = sys.argv[1]
if len(sys.argv) == 3:
    job_set = sys.argv[1]
    adopted_algo = sys.argv[2]

num_gpus_per_node, algo, dvfs_on, theta = adopted_algo.split('-')
num_gpus_per_node = int(num_gpus_per_node)
if dvfs_on == 'on':
    dvfs_on = True
else:
    dvfs_on = False
theta = float(theta)

num_gpus=NUM_GPUS
num_nodes=num_gpus / num_gpus_per_node
CLUSTER = {"num_node":num_nodes, "num_gpu":num_gpus_per_node, "gpu_mem":8192, "cpu_mem":16384, "node_idle_power":0, "network_speed":100} # unit is MB. 

print("number of nodes:%d, number of gpus per node:%d." % (num_nodes, num_gpus_per_node))

#jobG = job_generator(job_set)
#jobG.random_generate()
jobS = job_scheduler(job_set, CLUSTER, adopted_algo)

jobS.schedule(algo=algo, dvfs_on=dvfs_on, theta=theta)
jobS.print_stat()

from __future__ import print_function
import os, glob, sys
from random import sample, randint
import json, yaml
from cluster import *
from job import sim_job
import numpy as np
import random
import logging
from job_master import *
from common import *

job_set = 'offline-0.2'
adopted_algo = 'edf+wf-on-1.0'
num_gpus_per_node = 1
if len(sys.argv) == 2:
    adopted_algo = sys.argv[1]
if len(sys.argv) == 3:
    job_set = sys.argv[1]
    adopted_algo = sys.argv[2]
if len(sys.argv) == 4:
    job_set = sys.argv[1]
    adopted_algo = sys.argv[2]
    num_gpus_per_node = int(sys.argv[3])

algo, dvfs_on, theta = adopted_algo.split('-')
if dvfs_on == 'on':
    dvfs_on = True
else:
    dvfs_on = False
theta = float(theta)

num_gpus=NUM_GPUS
num_nodes=num_gpus // num_gpus_per_node
CLUSTER = {"num_node":num_nodes, "num_gpu":num_gpus_per_node, "gpu_mem":8192, "cpu_mem":16384, "node_idle_power":0, "network_speed":100} # unit is MB.
cluster_conf = "%d-%d" % (num_gpus, num_gpus_per_node)

logF = "logs/%s_%s_%s.log" % (job_set, cluster_conf, adopted_algo)
hdlr = logging.FileHandler(logF)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 

logger.info("number of nodes:%d, number of gpus per node:%d." % (num_nodes, num_gpus_per_node))
set_type, task_util = job_set.split("-")
task_util = float(task_util)
logger.info("job set type:%s, task set utilization:%f." % (set_type, task_util))
logger.info("algo:%s, dvfs:%s, theta:%f." % (algo, dvfs_on, theta))

run_Es = []
idle_Es = []
turnon_Es = []
total_Es = []
iters = 100
for i in range(iters):
    
    jobS = job_scheduler("%s-%d" % (job_set, i), CLUSTER, adopted_algo)
    if set_type == "offline":
        jobS.fast_offline(algo=algo, dvfs_on=dvfs_on, theta=theta)
    else:
        jobS.schedule(algo=algo, dvfs_on=dvfs_on, theta=theta)
    run_E, idle_E, turnon_E, total_E = jobS.print_stat()
    run_Es.append(run_E)
    idle_Es.append(idle_E)
    turnon_Es.append(turnon_E)
    total_Es.append(total_E)

logger.info("Average Run energy (aver-min-max) is %f-%f-%f" % (np.mean(run_Es), min(run_Es), max(run_Es)))
logger.info("Average Idle energy (aver-min-max) is %f-%f-%f" % (np.mean(idle_Es), min(idle_Es), max(idle_Es)))
logger.info("Average Turn-on energy (aver-min-max) is %f-%f-%f" % (np.mean(turnon_Es), min(turnon_Es), max(turnon_Es)))
logger.info("Average Total energy (aver-min-max) is %f-%f-%f" % (np.mean(total_Es), min(total_Es), max(total_Es)))


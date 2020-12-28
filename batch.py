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
if len(sys.argv) == 2:
  job_set = sys.argv[1]

iters = 100
algos = ["edf+spt-off-1.0", "edf+wf-off-1.0", "edf+bf-off-1.0", "lpt+ff-off-1.0"]
ngpns = [1, 2, 4, 8, 16]
for i in range(iters):
  logger.info("Generating new task...")
  jobG = job_generator(job_set)
  jobG.random_generate()
 
  for al in algos:
    for ngpn in ngpns:
      algo, dvfs_on, theta = al.split('-')
      if dvfs_on == 'on':
          dvfs_on = True
      else:
          dvfs_on = False
      theta = float(theta)
      
      num_gpus=NUM_GPUS
      num_nodes=num_gpus // ngpn
      CLUSTER = {"num_node":num_nodes, "num_gpu":ngpn, "gpu_mem":8192, "cpu_mem":16384, "node_idle_power":0, "network_speed":100} # unit is MB.
      cluster_conf = "%d-%d" % (num_gpus, ngpn)
      
      logF = "logs/%s_%s_%s.log" % (job_set, cluster_conf, adopted_algo)
      hdlr = logging.FileHandler(logF)
      hdlr.setFormatter(formatter)
      logger.addHandler(hdlr) 
      
      logger.info("number of nodes:%d, number of gpus per node:%d." % (num_nodes, ngpn))
      set_type, task_util = job_set.split("-")
      task_util = float(task_util)
      logger.info("job set type:%s, task set utilization:%f." % (set_type, task_util))
      logger.info("algo:%s, dvfs:%s, theta:%f." % (algo, dvfs_on, theta))
      
      jobS = job_scheduler(job_set, CLUSTER, adopted_algo)
      jobS.schedule(algo=algo, dvfs_on=dvfs_on, theta=theta)
      run_E, idle_E, turnon_E, total_E = jobS.print_stat()

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

job_set = 'offline-0.4'
if len(sys.argv) == 2:
  job_set = sys.argv[1]

iters=12
for i in range(iters):
  jobG =job_generator("%s-%d" % (job_set, i))
  jobG.random_generate()

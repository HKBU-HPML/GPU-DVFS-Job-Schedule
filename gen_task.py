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

schedule_type = 'offline'
util = 0.4
mode = 'synthetic'

if len(sys.argv) == 2:
    schedule_type = sys.argv[1]
if len(sys.argv) == 3:
    schedule_type = sys.argv[1]
    util = sys.argv[2]
if len(sys.argv) == 4:
    schedule_type = sys.argv[1]
    util = sys.argv[2]
    mode = int(sys.argv[3])

jobG =job_generator("%s-%d-%s" % (schedule_type, util, mode))
if mode == 'synthetic':
    jobG.rand_gen()
else:
    jobG.load_apps()
    jobG.real_gen()

jobG.save()


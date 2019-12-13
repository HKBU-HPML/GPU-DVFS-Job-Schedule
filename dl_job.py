import numpy as np
import time
import json, yaml
import logging
from settings import logger
import os

class DLJob:

    def __init__(self, job_root='job_configs', job_set='job_set_1', job_id=0):

        self.job_id = job_id
        self.job_config=os.path.join(job_root, job_set, "job_%d.json"%job_id)
        self.schedule_config=os.path.join(job_root, job_set, "schedule_%d.json"%job_id)
        self.load_job_config(self.job_config)
        self.load_schedule_config(self.schedule_config)

    def load_job_config(self, job_config):

        with open(job_config, 'r') as f:
            #self.job_json = json.load(f)
            self.job_json = yaml.safe_load(f)

        # network training setting
        self.dnn = self.job_json["dnn"]
        self.model_size = self.job_json["model_size"]
        self.lr = self.job_json["lr"]
        self.batch_size = self.job_json["batch_size"]
        self.dataset = self.job_json["dataset"]
        self.data_dir = self.job_json["data_dir"]
        self.nworkers = self.job_json["nworkers"]
        self.nsteps_update = self.job_json["nsteps_update"]
        self.iters = self.job_json['iters']
        self.cuda = True if self.job_json['cuda_enabled'] == 1 else False

        # estimated forward/backward time
        self.fw_time = self.job_json['fw_time']
        self.bw_time = self.job_json['bw_time']

    def load_schedule_config(self, schedule_config):

        with open(schedule_config, 'r') as f:
            #self.job_json = json.load(f)
            self.job_json = yaml.safe_load(f)

        self.device_ids = self.job_json['gpus']
        if self.cuda:
            self.ngpus = len(self.device_ids)
        else:
            self.ngpus = 0
        self.hostfile = self.job_json['hostfile']
        self.schedule = self.job_json['schedule']

    def get_forward_schedule(self, rank, iter_num):
        return self.schedule['rank_%d' % rank]['forward'][iter_num]

    def get_backward_schedule(self, rank, iter_num):
        return self.schedule['rank_%d' % rank]['backward'][iter_num]

    def get_communication_schedule(self, rank, iter_num):
        return self.schedule['rank_%d' % rank]['comm'][iter_num]

    def get_device(self, rank):
        return self.device_ids[rank]

    
    

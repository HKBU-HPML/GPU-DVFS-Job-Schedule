from __future__ import print_function
import os, glob, sys
from random import sample, randint
import json, yaml
from cluster import *
from common import *
from job import sim_job
import numpy as np
import random
import math
import pickle

NUM_GPUS = 2048

class job_generator:

    def __init__(self, set_name):
        self.set_name = set_name
        self.set_type, self.U, _ = set_name.split("-")
        self.U = float(self.U)
        self.num_jobs = int(NUM_GPUS * self.U)

        self.jobs = []

    def load_apps(self):
        f = open("apps.pkl", 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.apps = list(tmp_dict.values())

    def load(self):
        f = open("job_configs/%s.pkl" % self.set_name, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict) 
    
    def save(self):
        f = open("job_configs/%s.pkl" % self.set_name, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def show(self):
        for job in self.jobs:
            print(job)

    def get_jobs(self):
        return self.jobs

    def random_generate(self):

        job_id = 0
        actual_util = 0
        if self.set_type == "offline":
            for i in range(self.num_jobs):
                job_json = {}
                job_json["job_id"] = job_id
                job_json["job_name"] = "j%d" % job_id

                #job_json["D"] = random.uniform(10, 50)
                #job_json["delta"] = random.uniform(0.07, 0.91)
                #job_json["t0"] = random.randint(10, 100)
                #job_json["power_basic"] = random.uniform(50, 100)
                #job_json["gamma"] = random.uniform(30, 70)
                #job_json["cg"] = random.uniform(60, 100)

                ext_coef = random.randint(10, 50)
                job_json["D"] = random.uniform(1.66, 7.61) * ext_coef
                job_json["t0"] = random.uniform(0.1, 0.95) * ext_coef
                job_json["delta"] = random.uniform(0.07, 0.91)
                p_star = random.randint(175, 206)
                job_json["power_basic"] = p_star * random.uniform(0.20, 0.41)
                job_json["gamma"] = p_star * random.uniform(0.1, 0.2)
                job_json["cg"] = p_star - job_json["power_basic"] - job_json["gamma"]

                ## gpu performance modeling with DVFS
                #t_star = random.randint(20, 30) * random.randint(1, 10) 
                #job_json["t0"] = math.ceil(t_star * random.uniform(0.06, 0.89))
                #job_json["delta"] = random.uniform(0.07, 0.91)
                #job_json["D"] = t_star - job_json["t0"]

                ## gpu power modeling with DVFS
                #p_star = random.randint(50, 150) * 3
                #job_json["power_basic"] = p_star * random.uniform(0.20, 0.41)
                #job_json["gamma"] = p_star * random.uniform(0.1, 0.2)
                #job_json["cg"] = p_star - job_json["power_basic"] - job_json["gamma"]
                
                #num_apps = len(self.apps)
                #app_id = random.randint(0, num_apps - 1)
                #ext_coef = random.randint(10, 50)
                #job_json["D"] = self.apps[app_id]["D"] * ext_coef
                #job_json["delta"] = self.apps[app_id]["delta"]
                #job_json["t0"] = self.apps[app_id]["t0"] * ext_coef
                #job_json["power_basic"] = self.apps[app_id]["p0"]
                #job_json["gamma"] = self.apps[app_id]["gamma"]
                #job_json["cg"] = self.apps[app_id]["cg"]

                # job metrics
                job_json["arrival"] = 0
                job_json["utilization"] = random.uniform(0.25, 0.75)
                actual_util += job_json["utilization"]

                self.jobs.append(job_json)
                job_id += 1

        else:
            while True:
                task_dist = list(np.random.poisson(self.num_jobs*1.0/1440, size=1440 - 1))
                if np.sum(task_dist) == self.num_jobs:
                     logger.info("Got %d jobs." % self.num_jobs)
                     break

            task_dist.insert(0, int(NUM_GPUS * 0.4))
            for idx, n in enumerate(task_dist):
                for i in range(n):
                    job_json = {}
                    job_json["job_id"] = job_id
                    job_json["job_name"] = "j%d" % job_id

                    ext_coef = random.randint(10, 50)
                    job_json["D"] = random.uniform(1.66, 7.61) * ext_coef
                    job_json["t0"] = random.uniform(0.1, 0.95) * ext_coef
                    job_json["delta"] = random.uniform(0.07, 0.91)
                    p_star = random.randint(175, 206) 
                    job_json["power_basic"] = p_star * random.uniform(0.20, 0.41)
                    job_json["gamma"] = p_star * random.uniform(0.1, 0.2)
                    job_json["cg"] = p_star - job_json["power_basic"] - job_json["gamma"]

                    #job_json["D"] = random.uniform(10, 50)
                    #job_json["delta"] = random.uniform(0.07, 0.91)
                    #job_json["t0"] = random.randint(10, 100)
                    #job_json["power_basic"] = random.uniform(50, 100)
                    #job_json["gamma"] = random.uniform(30, 70)
                    #job_json["cg"] = random.uniform(60, 100)

                    ## gpu performance modeling with DVFS
                    #t_star = random.randint(100, 150)  
                    #job_json["t0"] = math.ceil(t_star * random.uniform(0.06, 0.89))
                    #job_json["delta"] = random.uniform(0.07, 0.91)
                    #job_json["D"] = t_star - job_json["t0"]

                    ## gpu power modeling with DVFS
                    #p_star = random.randint(200, 300)
                    #job_json["power_basic"] = p_star * random.uniform(0.20, 0.41)
                    #job_json["gamma"] = p_star * random.uniform(0.1, 0.2)
                    #job_json["cg"] = p_star - job_json["power_basic"] - job_json["gamma"]

                    # gpu performance modeling with DVFS, gtx1080ti
                    #num_apps = len(self.apps)
                    #app_id = random.randint(0, num_apps - 1)
                    #ext_coef = random.randint(10, 50)
                    #job_json["D"] = self.apps[app_id]["D"] * ext_coef
                    #job_json["delta"] = self.apps[app_id]["delta"]
                    #job_json["t0"] = self.apps[app_id]["t0"] * ext_coef
                    #job_json["power_basic"] = self.apps[app_id]["p0"]
                    #job_json["gamma"] = self.apps[app_id]["gamma"]
                    #job_json["cg"] = self.apps[app_id]["cg"]

                    #ext_coef = random.randint(15, 45)
                    #job_json["t0"] = random.uniform(0.0, 0.95) * ext_coef
                    #job_json["D"] = random.uniform(1.66, 7.61) * ext_coef
                    #job_json["delta"] = random.uniform(0.07, 0.91)

                    ## gpu power modeling with DVFS, gtx1080ti
                    #p_star = random.randint(175, 206)
                    #job_json["power_basic"] = p_star * random.uniform(0.20, 0.41)
                    #job_json["gamma"] = p_star * random.uniform(0.1, 0.2)
                    #job_json["cg"] = p_star - job_json["power_basic"] - job_json["gamma"]
                    #job_json["power_basic"] = random.randint(87, 118)
                    #job_json["gamma"] = random.randint(60, 89)
                    #job_json["cg"] = random.uniform(11, 17)

                    # job metrics
                    job_json["arrival"] = idx
                    job_json["utilization"] = random.uniform(0.15, 0.85)
                    if idx > 0:
                        actual_util += job_json["utilization"]

                    self.jobs.append(job_json)
                    job_id += 1

        logger.info("U_J is %f." % (actual_util / (0.5*NUM_GPUS)))

            
class job_scheduler:

    def __init__(self, set_name, cluster_dict, schedule_conf):
        self.job_root = "job_configs/%s" % set_name
        self.set_name = set_name
        self.schedule_conf = schedule_conf

        # statistical variable
        self.total_time = 0
        self.task_dist = []
        self.turn_on_dist = []

        self.clust = cluster(cluster_dict)

        if "online" in self.set_name:
            self.ARRIVAL_MAX = 1440
        else:
            self.ARRIVAL_MAX = 1

        self.job_set = [[] for i in range(self.ARRIVAL_MAX)] # simulate one day of 1440 minutes
        self.load()

    #def print_jobs(self):

    #    def job_format(job):
    #        return "job %d:\t%s-%s, %d-gpu, %d-iter." % (job.job_id, job.dnn, job.dataset, job.nworkers, job.iters)

    #    for job in self.job_set:
    #        logger.info(job_format(job))
    #    logger.info("")

    def load(self):
        jobG = job_generator(self.set_name)
        jobG.load()
        jobs = jobG.get_jobs()

        self.num_jobs = len(jobs)

        for job_json in jobs:
            self.job_set[job_json["arrival"]].append(sim_job(job_json))

        for i in range(self.ARRIVAL_MAX):
            self.task_dist.append(len(self.job_set[i]))

    def check_finished(self):
        finished_ids = []
        for i in range(self.ARRIVAL_MAX):
            for job in self.job_set[i]:
                if job.is_finished:
                    finished_ids.append(job.job_id)

        return finished_ids

    def fast_offline(self, algo="edf+spt", dvfs_on=True, theta=1.0):
        
        self.algo = algo
        self.pj_algo, self.pg_algo = algo.split("+")
        self.dvfs_on = dvfs_on
        self.theta = theta
        
        # get turn-on nodes
        on_nodes = self.clust.get_on_nodes()
        arrival_jobs = self.job_set[0]
        print_log = ""

        # solve dvfs
        if dvfs_on:
            for job in arrival_jobs:
                job.solve_dvfs()
        
        # EDF algorithm 
        if self.pj_algo == "edf":
            arrival_jobs = sorted(arrival_jobs, key=lambda x:(x.deadline))
        elif self.pj_algo == "lpt":
            arrival_jobs = sorted(arrival_jobs, key=lambda x:(-x.t_hat))

        for job in arrival_jobs:
 
            # get available gpus
            avail_gpus = []
            for node in on_nodes:
                avail_gpus.extend(node.gpu_list)

            found = False
            if len(avail_gpus) != 0:
                if self.pg_algo == "spt":
                    chosen_gpu = sorted(avail_gpus, key=lambda x:(x.end_time))[0]

                    if (job.deadline - chosen_gpu.end_time) >= job.t_hat:
                        chosen_gpu.add_job(job, 0)
                        found = True
                    else:
                        t_theta = job.get_t_theta(theta)
                        if dvfs_on and ((job.deadline - chosen_gpu.end_time) > t_theta):
                            job.theta_adjust(job.deadline - chosen_gpu.end_time)
                            chosen_gpu.add_job(job, 0)
                            found = True

                elif self.pg_algo == "bf":
                    avail_gpus = [gpu for gpu in avail_gpus if (gpu.end_time + job.t_hat) <= job.deadline]
                    if len(avail_gpus) != 0:
                        chosen_gpu = sorted(avail_gpus, key=lambda x:(x.max_load))[-1]
                        chosen_gpu.add_job(job, 0)
                        found = True
                    
                elif self.pg_algo == "wf":
                    avail_gpus = [gpu for gpu in avail_gpus if (gpu.end_time + job.t_hat) <= job.deadline]
                    if len(avail_gpus) != 0:
                        chosen_gpu = sorted(avail_gpus, key=lambda x:(x.max_load))[0]
                        chosen_gpu.add_job(job, 0)
                        found = True
                    
                elif self.pg_algo == "ff":
                    for gpu in avail_gpus:
                        if (job.deadline - gpu.end_time) >= job.t_hat:
                            chosen_gpu = gpu
                            chosen_gpu.add_job(job, 0)
                            found = True
                            break

	    	    
            if not found:
                # obtain a new node
                new_node = self.clust.get_off_nodes()
                new_node.turn_on()
                print_log += "turn on node %d.\n" % new_node.node_id
                chosen_gpu = new_node.gpu_list[0]
                chosen_gpu.add_job(job, 0)
                on_nodes.append(new_node)

            print_log += "node %d-gpu %d: running job-%d(job_time = %f, ddl = %f, end_time = %f).\n" % (chosen_gpu.node_id, chosen_gpu.gpu_id, job.job_id, job.t_hat, job.deadline, job.finish_time)

        #if print_log != "":
        #    logger.info("Time: %d\n%s" % (time, print_log))
 
        self.total_time = 0
        for node in self.clust.node_list:
            node.set_off_active_time()
            node.set_off_idle_energy()
            self.total_time = max(node.active_time, self.total_time)

    def schedule(self, algo="edl+spt", dvfs_on=True, theta=1.0):
        
        self.algo = algo
        self.pj_algo, self.pg_algo = algo.split("+")
        self.dvfs_on = dvfs_on
        self.theta = theta
        cur_time = 0
        
        job_id_pool = [i for i in range(self.num_jobs)]

        time = 0
        num_finished_jobs = 0
        while len(job_id_pool) != 0:

            print_log = ""

            # update the idle power if any, status of gpus and jobs
            for node in self.clust.node_list:
                node.update_idle_energy()
                node.update_status(time)

            # check if some jobs have been finished
            finished_job_ids = self.check_finished()
            if len(finished_job_ids) > num_finished_jobs:
                num_finished_jobs = len(finished_job_ids)
                print_log += "finished: %s\n" % finished_job_ids
                print_log += "finished: %d\n" % len(finished_job_ids)
                job_id_pool = [job_id for job_id in job_id_pool if job_id not in finished_job_ids]

            # use DRS to shut down some nodes
            for node in self.clust.node_list:
                if node.shutdown(drs_thres = 2):
                    print_log += "turn off node %d.\n" % node.node_id

            if time < self.ARRIVAL_MAX:
                arrival_jobs = self.job_set[time]
                if dvfs_on:
                    for job in arrival_jobs:
                        job.solve_dvfs()

                # get turn-on nodes
                on_nodes = self.clust.get_on_nodes()
                self.turn_on_dist.append(len(on_nodes))

                # solve offline deadline-prior jobs
                if (time == 0) and dvfs_on:
                    dp_jobs = [job for job in arrival_jobs if job.job_type == "dp"]
                    logger.info("The number of offline deadline-prior tasks is %d." % len(dp_jobs))
                    arrival_jobs = [job for job in arrival_jobs if job.job_type == "ep"]
                    
                    # needed node number
                    num_nodes = (len(dp_jobs) - 1) // self.clust.num_gpus_per_node + 1
                    selected_nodes = self.clust.node_list[:num_nodes]

                    job_idx = 0
                    for node in selected_nodes:
                        node.turn_on()
                        on_nodes.append(node)
                        print_log += "turn on node %d for offline tasks.\n" % node.node_id
                        for gpu in node.gpu_list:
                            if job_idx < len(dp_jobs):
                                job = dp_jobs[job_idx]
                                gpu.add_job(job, time)
                                print_log += "node %d-gpu %d: running job-%d(job_time = %f(%f), ddl = %f, end_time = %f).\n" % (gpu.node_id, gpu.gpu_id, job.job_id, job.t_hat, job.t_star, job.deadline, job.finish_time)
                                job_idx += 1
        
                # EDF algorithm 
                if self.pj_algo == "edf":
                    arrival_jobs = sorted(arrival_jobs, key=lambda x:(x.deadline))
                elif self.pj_algo == "lpt":
                    arrival_jobs = sorted(arrival_jobs, key=lambda x:(-x.t_hat))

                for job in arrival_jobs:
 
                    # get available gpus
                    avail_gpus = []
                    for node in on_nodes:
                        avail_gpus.extend(node.gpu_list)

                    found = False
                    if len(avail_gpus) != 0:
                        if self.pg_algo == "spt":
                            chosen_gpu = sorted(avail_gpus, key=lambda x:(x.end_time))[0]

                            if (job.deadline - max(time, chosen_gpu.end_time)) >= job.t_hat:
                                chosen_gpu.add_job(job, time)
                                found = True
                            else:
                                t_theta = job.get_t_theta(theta)
                                if dvfs_on and ((job.deadline - max(time, chosen_gpu.end_time)) > t_theta):
                                    job.theta_adjust(job.deadline - max(time, chosen_gpu.end_time))
                                    chosen_gpu.add_job(job, time)
                                    found = True

                        elif self.pg_algo == "bf":
                            avail_gpus = [gpu for gpu in avail_gpus if (gpu.end_time + job.t_hat) <= job.deadline]
                            if len(avail_gpus) != 0:
                                chosen_gpu = sorted(avail_gpus, key=lambda x:(x.max_load))[-1]
                                chosen_gpu.add_job(job, time)
                                found = True
                            
                        elif self.pg_algo == "wf":
                            avail_gpus = [gpu for gpu in avail_gpus if (gpu.end_time + job.t_hat) <= job.deadline]
                            if len(avail_gpus) != 0:
                                chosen_gpu = sorted(avail_gpus, key=lambda x:(x.max_load))[0]
                                chosen_gpu.add_job(job, time)
                                found = True
                            
                        elif self.pg_algo == "ff":
                            for gpu in avail_gpus:
                                if (job.deadline - max(time, gpu.end_time)) >= job.t_hat:
                                    chosen_gpu = gpu
                                    chosen_gpu.add_job(job, time)
                                    found = True
                                    break

                        elif self.pg_algo == "bin":
                            # online bin-packing algorithm, default
                            if time == 0:  # worst-fit
                                avail_gpus = [gpu for gpu in avail_gpus if (gpu.end_time + job.t_hat) <= job.deadline]
                                if len(avail_gpus) != 0:
                                    chosen_gpu = sorted(avail_gpus, key=lambda x:(x.max_load))[0]
                                    chosen_gpu.add_job(job, time)
                                    found = True
                            else:
                                for gpu in avail_gpus:
                                    if (job.deadline - max(time, gpu.end_time)) >= job.t_hat:
                                        chosen_gpu = gpu
                                        chosen_gpu.add_job(job, time)
                                        found = True
                                        break
	            	    
                    if not found:
                        # obtain a new node
                        new_node = self.clust.get_off_nodes()
                        new_node.turn_on()
                        print_log += "turn on node %d.\n" % new_node.node_id
                        chosen_gpu = new_node.gpu_list[0]
                        chosen_gpu.add_job(job, time)
                        on_nodes.append(new_node)

                    print_log += "node %d-gpu %d: running job-%d(job_time = %f(%f), ddl = %f, end_time = %f).\n" % (chosen_gpu.node_id, chosen_gpu.gpu_id, job.job_id, job.t_hat, job.t_star, job.deadline, job.finish_time)

            #if print_log != "":
            #    logger.info("Time: %d\n%s" % (time, print_log))

            time += 1
            #if time > 1400:
            #    break

        self.total_time = time
                
    def print_stat(self):

        idle_time = 0
        for node in self.clust.node_list:
            if node.active_time != 0:
                logger.info("node-%d: %d / %d." % (node.node_id, node.active_time, self.total_time))
                for gpu in node.gpu_list:
                    logger.info("\t gpu-%d: %d / %d." % (gpu.gpu_id, gpu.active_time, self.total_time))
                    idle_time += node.active_time - gpu.active_time
        logger.info("total idle time: %f." % idle_time)

        #aver_job_time = np.mean([j.finish_time for j in self.job_set])
        #print "Average Job Completion Time is %f ms." % aver_job_time

        logger.info("Algorithm %s with DVFS-%s-%f:" % (self.algo, self.dvfs_on, self.theta))
        logger.info("Run energy is %f." % self.clust.get_run_energy())
        logger.info("Idle energy is %f." % self.clust.get_idle_energy())
        logger.info("Turn-on energy is %f." % self.clust.get_turn_on_energy())
        logger.info("Total energy is %f." % self.clust.get_total_energy())

        #self.brief_log = "logs/brief/%s-%s.log" % (self.set_name, self.schedule_conf)
        #self.brief_log = "logs/brief/%s-%s-%s.log" % (self.set_name, self.algo, self.dvfs_on)
        #with open(self.brief_log, "w") as f:
        #    f.write("Algorithm %s with DVFS-%s-%f:\n" % (self.algo, self.dvfs_on, self.theta))
        #    f.write("Task Distribution:%s\n" % self.task_dist)
        #    f.write("Turn-on Node Distribution:%s\n" % self.turn_on_dist)
        #    f.write("Run energy:%f\n" % self.clust.get_run_energy())
        #    f.write("Idle energy:%f\n" % self.clust.get_idle_energy())
        #    f.write("Turn-on energy:%f\n" % self.clust.get_turn_on_energy())
        #    f.write("Total energy:%f\n" % self.clust.get_total_energy())

        return (self.clust.get_run_energy(), self.clust.get_idle_energy(), self.clust.get_turn_on_energy(), self.clust.get_total_energy())
            

    def write_allocate(self):

        def gpu_allocate(nworkers):
            #nodes = {
            #         "gpu10":[i % 4 for i in range(nworkers/2)], 
            #         "gpu11":[i % 4 for i in range(nworkers/2)], 
            #}
            nodes = {
                     "localhost":[-1 for i in range(nworkers)], 
            }
            return nodes
            
        for idx, job in enumerate(self.job_set):

            job_json = job.job_conf
            schedule = job_json.copy()

            # allocate nodes and GPUs
            node_gpu = gpu_allocate(job_json['nworkers'])
            hostfile = os.path.join(self.job_root, "cluster_j%d" % job_json['job_id'])
            schedule["hostfile"] = hostfile
            schedule["gpus"] = []
            with open(hostfile, "w") as f:
                for node in node_gpu:
                    f.write("%s slots=%d\n" % (node, len(node_gpu[node])))
                    schedule["gpus"].extend(node_gpu[node])
            
            # schedule the tasks
            schedule["schedule"] = {}
            for r in range(job_json['nworkers']):
                tmp_plan = {}
                f = []
                b = []
                c = []
                for i in range(job_json['iters']):
                    f.append(0)
                    b.append(0)
                    c.append(0)
                tmp_plan["forward"] = f
                tmp_plan["backward"] = b
                tmp_plan["comm"] = c
                schedule["schedule"]["rank_%d"%r] = tmp_plan
       
            with open(os.path.join(self.job_root, "schedule_%d.json"%idx), "w") as f:
                yaml.safe_dump(schedule, f)

    def write_schedule(self):
        pass


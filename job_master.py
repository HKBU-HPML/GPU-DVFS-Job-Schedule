import os, glob, sys
from random import sample, randint
import json, yaml
from cluster import *
from job import sim_job
import numpy as np
import random

adopted_algo = '1-bp-false-1.0'
if len(sys.argv) == 2:
    adopted_algo = sys.argv[1]
num_gpus_per_node, algo, dvfs_on, theta = adopted_algo.split('-')
num_gpus_per_node = int(num_gpus_per_node)
if dvfs_on == 'true':
    dvfs_on = True
else:
    dvfs_on = False
theta = float(theta)

ARRIVAL_MAX=1440
U_ON=1.6
U_OFF=0.4
num_gpus=2048
num_nodes=num_gpus / num_gpus_per_node
CLUSTER = {"num_node":num_nodes, "num_gpu":num_gpus_per_node, "gpu_mem":8192, "cpu_mem":16384, "node_idle_power":0, "network_speed":100} # unit is MB. 

print "number of nodes:%d, number of gpus per node:%d." % (num_nodes, num_gpus_per_node)

class job_generator:

    def __init__(self, set_name):
        self.set_name = set_name
        self.num_jobs = int(num_gpus * U_ON)

    def random_generate(self):

        while True:
            task_dist = np.random.poisson(self.num_jobs*1.0/ARRIVAL_MAX, size=ARRIVAL_MAX)
            if np.sum(task_dist) == self.num_jobs:
                 print "Got %d jobs." % self.num_jobs
                 break

        self.job_root = "job_configs/%s_%d" % (self.set_name, self.num_jobs)
        if not os.path.exists(self.job_root):
            os.makedirs(self.job_root)

        job_id = 0
        actual_util = 0
        for idx, n in enumerate(task_dist):
            for i in range(n):
                job_json = {}
                job_json["job_id"] = job_id
                job_json["job_name"] = "j%d" % job_id

                # gpu performance modeling with DVFS
                job_json["D"] = random.uniform(10, 40)
                job_json["delta"] = random.uniform(0.1, 0.9)
                job_json["t0"] = random.randint(10, 20)

                # gpu power modeling with DVFS
                job_json["power_basic"] = random.uniform(50, 100)
                job_json["gamma"] = random.uniform(30, 70)
                job_json["cg"] = random.uniform(60, 100)

                # job metrics
                job_json["arrival"] = idx
                job_json["utilization"] = random.uniform(0.1, 0.9)
                actual_util += job_json["utilization"]

                with open(os.path.join(self.job_root, "job_%d.json"%job_id), "w") as f:
                    yaml.safe_dump(job_json, f)

                job_id += 1

        print "U_J is %f." % (actual_util / (0.5*num_gpus))

            
class job_scheduler:

    def __init__(self, set_name):
        self.job_root = "job_configs/%s" % set_name
        self.set_name = set_name
        self.job_files = glob.glob(r'job_configs/%s/job*.json' % set_name)
        self.num_jobs = len(self.job_files)

        # statistical variable
        self.total_time = 0
        self.task_dist = []
        self.turn_on_dist = []

        self.job_set = [[] for i in range(ARRIVAL_MAX)] # simulate one day of 1440 minutes
        self.load_job_set()

        self.clust = cluster(CLUSTER)


    def print_jobs(self):

        def job_format(job):
            return "job %d:\t%s-%s, %d-gpu, %d-iter." % (job.job_id, job.dnn, job.dataset, job.nworkers, job.iters)

        for job in self.job_set:
            print job_format(job)
        print ""

    def load_job_set(self):
        for jf in self.job_files:
            with open(jf, 'r') as f:
                job_json = yaml.safe_load(f)
                self.job_set[job_json["arrival"]].append(sim_job(job_json))

        for i in range(ARRIVAL_MAX):
            self.task_dist.append(len(self.job_set[i]))

    def check_finished(self):
        finished_ids = []
        for i in range(ARRIVAL_MAX):
            for job in self.job_set[i]:
                if job.is_finished:
                    finished_ids.append(job.job_id)

        return finished_ids

    def schedule(self, algo="edl", dvfs_on=True, theta=1.0):
        
        self.algo = algo
        self.dvfs_on = dvfs_on
        self.theta = theta
        cur_time = 0
        
        job_id_pool = [i for i in range(self.num_jobs)]

        time = 0
        num_finished_jobs = 0
        while len(job_id_pool) != 0:

            print_log = ""

            # update status of gpus and jobs
            for node in self.clust.node_list:
                node.update_idle_energy()
                node.update_status(time)

            # check if some jobs have been finished
            finished_job_ids = self.check_finished()
            if len(finished_job_ids) > num_finished_jobs:
                num_finished_jobs = len(finished_job_ids)
                # print_log += "finished: %s\n" % finished_job_ids
                print_log += "finished: %d\n" % len(finished_job_ids)
                job_id_pool = [job_id for job_id in job_id_pool if job_id not in finished_job_ids]

            # use DRS to shut down some nodes
            for node in self.clust.node_list:
                if node.shutdown(drs_thres = 2):
                    print_log += "turn off node %d.\n" % node.node_id

            if time >= ARRIVAL_MAX:
                if print_log != "":
                    print "Time: %d\n%s" % (time, print_log)
                time += 1
                continue

            arrival_jobs = self.job_set[time]
            if dvfs_on:
                for job in arrival_jobs:
                    job.solve_dvfs()
            arrival_jobs = sorted(arrival_jobs, key=lambda x:(x.deadline))

            # get turn-on nodes
            on_nodes = self.clust.get_on_nodes()
            self.turn_on_dist.append(len(on_nodes))
            for job in arrival_jobs:
 
                # get available gpus
                avail_gpus = []
                for node in on_nodes:
                    avail_gpus.extend(node.gpu_list)

                if len(avail_gpus) != 0:
                    # EDL algorithm 
                    if algo == "edl":
                        chosen_gpu = sorted(avail_gpus, key=lambda x:(x.end_time))[0]

                        if (job.deadline - max(time, chosen_gpu.end_time)) >= job.t_hat:
                            chosen_gpu.add_job(job, time)
                        else:
                            t_theta = job.get_t_theta(theta)
                            if dvfs_on and ((job.deadline - max(time, chosen_gpu.end_time)) > t_theta):
                                job.theta_adjust(job.deadline - max(time, chosen_gpu.end_time))
                                chosen_gpu.add_job(job, time)
                            else:
                                # obtain a new node
                                new_node = self.clust.get_off_nodes()
                                new_node.turn_on()
                                print "turn on node %d.\n" % new_node.node_id
                                chosen_gpu = new_node.gpu_list[0]
                                chosen_gpu.add_job(job, time)
                                on_nodes.append(new_node)

                    else:
                        # bin-packing algorithm, default
                        is_found = False
                        for gpu in avail_gpus:
                            if (job.deadline - max(time, gpu.end_time)) >= job.t_hat:
                                chosen_gpu = gpu
                                chosen_gpu.add_job(job, time)
                                is_found = True
                                break
                        if not is_found:
                            # obtain a new node
                            new_node = self.clust.get_off_nodes()
                            new_node.turn_on()
                            print "turn on node %d.\n" % new_node.node_id
                            chosen_gpu = new_node.gpu_list[0]
                            chosen_gpu.add_job(job, time)
                            on_nodes.append(new_node)
                      
                else:
                    # obtain a new node
                    new_node = self.clust.get_off_nodes()
                    new_node.turn_on()
                    print "turn on node %d.\n" % new_node.node_id
                    chosen_gpu = new_node.gpu_list[0]
                    chosen_gpu.add_job(job, time)
                    on_nodes.append(new_node)

                print_log += "node %d-gpu %d: running job-%d(job_time = %f, ddl = %f, end_time = %f).\n" % (chosen_gpu.node_id, chosen_gpu.gpu_id, job.job_id, job.t_hat, job.deadline, job.finish_time)

            if print_log != "":
                print "Time: %d\n%s" % (time, print_log)

            time += 1
            #if time > 1400:
            #    break

        self.total_time = time

                
    def print_stat(self):

        for node in self.clust.node_list:
            if node.active_time != 0:
                print "node-%d: %d / %d." % (node.node_id, node.active_time, self.total_time)
                for gpu in node.gpu_list:
                    print "\t gpu-%d: %d / %d." % (gpu.gpu_id, gpu.active_time, self.total_time)

        #aver_job_time = np.mean([j.finish_time for j in self.job_set])
        #print "Average Job Completion Time is %f ms." % aver_job_time

        print "Algorithm %s with DVFS-%s-%f:" % (self.algo, self.dvfs_on, self.theta)
        print "Run energy is %f." % self.clust.get_run_energy()
        print "Idle energy is %f." % self.clust.get_idle_energy()
        print "Turn-on energy is %f." % self.clust.get_turn_on_energy()
        print "Total energy is %f." % self.clust.get_total_energy()

        self.brief_log = "logs/brief/%s-%s.log" % (self.set_name, adopted_algo)
        #self.brief_log = "logs/brief/%s-%s-%s.log" % (self.set_name, self.algo, self.dvfs_on)
        with open(self.brief_log, "w") as f:
            f.write("Algorithm %s with DVFS-%s-%f:\n" % (self.algo, self.dvfs_on, self.theta))
            f.write("Task Distribution:%s\n" % self.task_dist)
            f.write("Turn-on Node Distribution:%s\n" % self.turn_on_dist)
            f.write("Run energy:%f\n" % self.clust.get_run_energy())
            f.write("Idle energy:%f\n" % self.clust.get_idle_energy())
            f.write("Turn-on energy:%f\n" % self.clust.get_turn_on_energy())
            f.write("Total energy:%f\n" % self.clust.get_total_energy())
            

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

jobG = job_generator("online_dvfs")
jobG.random_generate()
jobS = job_scheduler("online_dvfs")

jobS.schedule(algo=algo, dvfs_on=dvfs_on, theta=theta)
jobS.print_stat()


class gpu:

    def __init__(self, mem, gpu_id, host_node):
        
        self.gpu_mem = mem
        self.gpu_idle_power = 85
        self.gpu_id = gpu_id
        self.host_node = host_node
        self.node_id = self.host_node.node_id
        self.job_list = []

        self.accum_task_time = 0
        self.loads = []
        self.max_load = 0
        self.cur_job = ""

        self.allocated_mem = 0
        self.free_mem = self.gpu_mem
        self.end_time = 0

        self.is_busy = False

        # variable for scheduling
        self.active_time = 0

        self.idle_energy = 0
        self.run_energy = 0

    # allocate stage
    def add_job(self, job, time):
        start_time = max(self.end_time, time)
        self.end_time = start_time + job.t_hat
        job.set_finish_time(self.end_time)
        self.job_list.append(job)
        self.run_energy += job.t_hat * job.p_hat

        # set the GPU load
        self.accum_task_time += job.t_hat
        self.loads.append(self.accum_task_time / job.deadline)
        self.max_load = max(self.loads)

        if self.cur_job == "":
            self.cur_job = self.job_list.pop(0)
            self.is_busy = True

    def update_status(self, time):
        # update statistical data
        cur_util = 0
        if self.is_busy == True:
            cur_util = 1
            self.active_time += 1

        if self.cur_job != "":
            if time >= self.cur_job.finish_time:
                self.cur_job.is_finished = True
                self.cur_job = ""
                if len(self.job_list) != 0:
                    self.cur_job = self.job_list.pop(0)
                else:
                    self.is_busy = False

        return cur_util

    def update_idle_energy(self):

        if not self.is_busy:
            self.idle_energy += self.gpu_idle_power
    
class node:

    def __init__(self, cpu_mem, num_gpu, net_spd, node_id):

        self.cpu_mem = cpu_mem
        self.num_gpu = num_gpu
        self.net_spd = net_spd
        self.node_id = node_id

        self.gpu_list = [gpu(mem=8192, gpu_id=i, host_node=self) for i in range(self.num_gpu)]
        self.event_start_time = []
        self.event_end_time = []
        self.job_list = []

        self.compute_load = 0

        self.net_conf = {"full_speed": 128.0,
                         "alpha": 0.0,
                         "beta": 1000.0 / 128.0,
                         "eta": 0.7,
                         "num_of_task": 0}

        # record the variable for scheduling
        self.active_time = 0
        self.makespan = 0

        # power relative
        self.status = "off"
        self.drs_wait = 0
        self.turn_on_overhead = 200
        self.node_power = 0

        # energy statistic
        self.turn_on_energy = 0

    def get_idle_energy(self):
        self.idle_energy = 0
        for gpu in self.gpu_list:
            self.idle_energy += gpu.idle_energy
        return self.idle_energy

    def get_run_energy(self):
        self.run_energy = 0
        for gpu in self.gpu_list:
            self.run_energy += gpu.run_energy
        return self.run_energy

    def get_turn_on_energy(self):
        return self.turn_on_energy

    def get_total_energy(self):
        self.total_energy = 0
        for gpu in self.gpu_list:
            self.total_energy += gpu.idle_energy + gpu.run_energy
        self.total_energy += self.turn_on_energy
        return self.total_energy

    def update_idle_energy(self):

        if self.status == "off":
            return

        for gpu in self.gpu_list:
            gpu.update_idle_energy()

    def update_status(self, time):
        cur_utils = 0
        for gpu in self.gpu_list:
            cur_utils += gpu.update_status(time)
        if cur_utils != 0:
            self.active_time += 1
            self.drs_wait = 0
        else:
            if self.status == "on":
                self.drs_wait += 1
 
    def shutdown(self, drs_thres = 5):
        if self.status == "on" and self.drs_wait >= drs_thres:
            self.status = "off"
            self.drs_wait = 0
            return True
        else:
            return False

    def turn_on(self):
        self.status = "on"
        self.drs_wait = 0
        self.turn_on_energy += self.turn_on_overhead

    def print_jobs(self):
        job_ids = [job.job_id for job in self.job_list]
        print self.node_id, job_ids

        for gpu in self.gpu_list:
            gpu_job_ids = [job.job_id for job in gpu.job_list]
            print "\t", gpu.gpu_id, gpu_job_ids, gpu.wk_id_list

    def update(self):
        
        # update network workload
        self.net_load = 0
        for job in self.job_list:
            self.net_load += job.model_size

        # update compute workload
        self.compute_load = 0
        for gpu in self.gpu_list:
            self.compute_load += gpu.workload


class cluster:

    def __init__(self, config): # config is a dict

        self.num_node = config["num_node"]
        self.node_list = [node(config["cpu_mem"], config["num_gpu"], config["network_speed"], i) for i in range(self.num_node)]
        self.num_gpus_per_node = config["num_gpu"]

        self.gpu_list = []
        for n in self.node_list:
            self.gpu_list.extend(n.gpu_list)

    def update(self):

        for node in self.node_list:
            node.update()
        
    def get_on_nodes(self):

        on_nodes = []
        for node in self.node_list:
            if node.status == "on":
                on_nodes.append(node)

        return on_nodes

    def get_off_nodes(self):

        for node in self.node_list:
            if node.status == "off":
                return node

    def get_total_energy(self):
        total_energy = 0
        for node in self.node_list:
            total_energy += node.get_total_energy()
        return total_energy

    def get_idle_energy(self):
        idle_energy = 0
        for node in self.node_list:
            idle_energy += node.get_idle_energy()
        return idle_energy

    def get_turn_on_energy(self):
        turn_on_energy = 0
        for node in self.node_list:
            turn_on_energy += node.get_turn_on_energy()
        return turn_on_energy

    def get_run_energy(self):
        run_energy = 0
        for node in self.node_list:
            run_energy += node.get_run_energy()
        return run_energy

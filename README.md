# Energy-aware Non-Preemptive Task Scheduling with Deadline Constraint in DVFS-enabled Heterogeneous Clusters

This repository contains the code for modeling/benchmarking NVIDIA GPU performance and power with dynamic voltage and frequency scaling. The relevant papers are as follows:
+ X. Mei, Q. Wang, H. Liu, Y.-W. Leung, Z. Li and X. Chu, "Energy-aware Non-Preemptive Task Scheduling with Deadline Constraint in DVFS-enabled Heterogeneous Clusters," IEEE Transactions on Parallel and Distributed Systems.[under review]

### Citation
```
@article{mei2021energy,
  title={Energy-aware Task Scheduling with Deadline Constraint in DVFS-enabled Heterogeneous Clusters},
  author={Mei, Xinxin and Wang, Qiang and Liu, Hai and Leung, Yiu-Wing and Li, Zongpeng and Chu, Xiaowen},
  journal={arXiv preprint arXiv:2104.00486},
  year={2021}
}
```

## Content
1. Introduction
2. Usage
3. Results
4. Contacts

## Usage
### Dependencies and prerequisites
+ Python 3.6+
+ CUDA 10.0 or above
+ NVIDIA GPU Driver (the latest version is recommended.)
+ OS requirement: Windows 7/10/11, Ubuntu 16.04 or above, CentOS 6/7.
+ Using "pip install -r requirements.txt" to install the required python libraries.

Tips: If you are using Windows, we recommend you to use WSL to install a linux sub-system (e.g., Ubuntu), and run the scripts. 

### Performance Modeling with DVFS

The GPU configurations are as follows. We collect the performance and power data of two GPU devices. 
GPU|Base Core Frequency (MHz)|Base Memory Frequency (MHz)
:--|:--:|:--:
GTX 2070 SUPER | 1880 | 6300 
GTX 1080 Ti | 1800 | 5000

1. Firstly please set the GPU variables in [cluster.py](https://github.com/HKBU-HPML/GPU-DVFS-Job-Schedule/blob/master/cluster.py), including:
```
GPU_NAME = 'gtx2070s'
CORE_BASE = 1880
MEM_BASE = 6300
```
The benchmarking results are stored in [./csvs](https://github.com/HKBU-HPML/GPU-DVFS-Job-Schedule/tree/master/csvs), one csv file per GPU. 

2. Fit the parameters of performance and power for each benchmark.
```
python model.py
```
Then the fitting results will be printed out and "apps.pkl" which stores the results will be generated.

### Simulation
1. Generate the task set.
```
python gen_task.py ${type} ${util} ${mode}
```
${type} can be "offline" or "online". ${util} is a float number to define the average GPU utilization. ${mode} can be "synthetic" or "real". For example, to generate an offline task set with U=1.0, in which the tasks are from the real benchmarking data, one can run
```
python gen_task.py offline 1.0 real
```
The task set file will be generated in the folder [./job_configs](https://github.com/HKBU-HPML/GPU-DVFS-Job-Schedule/tree/master/job_configs).

2. Run the simulation for a task set.
```
python main.py ${type}-${util}-{mode} ${algo}-${dvfs}-${theta} ${gpn}
```
${type} can be "offline" or "online". ${util} is a float number to define the average GPU utilization. ${mode} can be "synthetic" or "real".
${algo} indicates the algorithm chosen to schedule the tasks. ${dvfs} can be "on" or "off", indicating that if the dvfs optimization is activated. ${gpn} means gpus per node, and is an integer.

### Real Experiments

## Contact
Email: [qiangwang@comp.hkbu.edu.hk](mainto:qiangwang@comp.hkbu.edu.hk)

Personal Website: [https://blackjack2015.github.io](https://blackjack2015.github.io)

Welcome any suggestion or concern!

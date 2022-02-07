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
GPU|Base Core Frequency (MHz)|Base Memory Frequency (MHz)
:--|:--:|:--:
GTX 2070 SUPER | 1880 | 6300 
GTX 1080 Ti | 1800 | 5000

### Simulation


### Real Experiments

## Contact
Email: [qiangwang@comp.hkbu.edu.hk](mainto:qiangwang@comp.hkbu.edu.hk)

Personal Website: [https://blackjack2015.github.io](https://blackjack2015.github.io)

Welcome any suggestion or concern!

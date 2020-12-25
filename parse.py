import glob

logRoot = 'logs/v2'
fns = glob.glob(r'%s/*.log' % logRoot)

with open("csvs/results.csv", "w") as f:
    f.write("gpus,gpus_per_node,util,algo,dvfs_on,theta,aver_run_E,min_run_E,max_run_E,aver_idle_E,min_idle_E,max_idle_E,aver_turnon_E,min_turnon_E,max_turnon_E,aver_total_E,min_total_E,max_total_E\n")
    for fn in fns:
        print(fn)
        task_set, cluster_conf, algo_conf = fn[:-4].split("_")
        _, util = task_set.split("-")
        gpus, gpus_per_node = cluster_conf.split("-")
        algo, dvfs_on, theta = algo_conf.split("-")
        if dvfs_on == "on":
            dvfs_on = 1
        else:
            dvfs_on = 0
        with open(fn, "r") as fl:
            contents = fl.readlines()
            aver_run_E, min_run_E, max_run_E = contents[-4].split()[-1].split("-")
            aver_idle_E, min_idle_E, max_idle_E = contents[-3].split()[-1].split("-")
            aver_turnon_E, min_turnon_E, max_turnon_E = contents[-2].split()[-1].split("-")
            aver_total_E, min_total_E, max_total_E = contents[-1].split()[-1].split("-")

        vals = (512,int(gpus_per_node),float(util),
		algo, dvfs_on, float(theta),
		float(aver_run_E), float(min_run_E), float(max_run_E),
                float(aver_idle_E), float(min_idle_E), float(max_idle_E),
                float(aver_turnon_E), float(min_turnon_E), float(max_turnon_E),
                float(aver_total_E), float(min_total_E), float(max_total_E))
        print(vals)
        f.write(",".join([str(val) for val in vals]))
        f.write("\n")
					
        

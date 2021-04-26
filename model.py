import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Lasso

CORE_BASE = 1800
MEM_BASE = 5000

def get_ci(samples):

    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    z = 1.96 # 95%
    se = std /np.sqrt(len(samples))

    lcb = mean - z * se
    ucb = mean + z * se

    return lcb, ucb

def LR(X, y, alpha = 0.05):    
    
    #lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000, positive=True, random_state=9999, selection='random', fit_intercept=False)
    lin = Lasso(alpha=0.01, positive=True, fit_intercept=False)
    lin.fit(X,y)
    print(lin.coef_, lin.intercept_)
    w = lin.coef_
    #w = np.array([lin.intercept_, lin.coef_[0], lin.coef_[1]])
    print(w)

    #inv_item = np.linalg.inv((np.dot(X.T, X)) + alpha * np.identity(X.shape[1]))
    #w = np.dot((inv_item @ X.T), y)

    return w

df = pd.read_csv("csvs/gtx1080ti-dvfs-real-Performance-Power.csv", header = 0)
print(df.columns)

#extras = ['backpropBackward', 'binomialOptions', 'cfd', 'eigenvalues', 'gaussian', 'srad', 'dxtc', 'pathfinder', 'scanScanExclusiveShared', 'stereoDisparity'] 
#df = df[~df.appName.isin(extras)]

kernels = df['appName'].drop_duplicates()
kernels.sort_values(inplace=True)

para_dict = {}
vp = 0.5
time_err = 0
power_err = 0
p_basic = []
gamma = []
cg = []
for kernel in kernels:

    tmp_df = df[df.appName == kernel]

    coreFs = tmp_df["coreF"].values * 1.0 / CORE_BASE
    coreVs = (coreFs - vp) ** 2 * 2 + vp
    memFs = tmp_df["memF"].values * 1.0 / MEM_BASE

    # fitting performance
    x1 = 1 / coreFs
    x2 = 1 / memFs
    X = np.stack((np.ones(x1.shape[0]), x1, x2)).T
    y = tmp_df["time/ms"].values
    print(X, y)
    tw = LR(X, y)
    t0 = tw[0]
    D = tw[1] + tw[2]
    delta = tw[1] / D
    print("t0: %f, D: %f, delta: %f." % (t0, D, delta))
    rec_y = D * (delta * x1 + (1 - delta) * x2) + t0
    #rec_y = tw.predict(X)
    time_err += np.mean((y - rec_y) ** 2)

    # fitting power 
    VFs = coreVs ** 2 * coreFs
    X = np.stack((np.ones(memFs.shape[0]), memFs, VFs)).T
    y = tmp_df["power/W"].values
    print(X, y)
    pw = LR(X, y)
    p0 = pw[0]
    gamma = pw[1]
    cg = pw[2]
    rec_y = p0 + gamma*memFs + cg*VFs
    #rec_y = pw.predict(X)
    power_err += np.mean((y - rec_y) ** 2)

    para_dict[kernel] = {"p_star":(p0+gamma+cg), "p0":p0, "gamma":gamma, "cg":cg, "t_star":(t0+D), "t0":t0, "D":D, "delta":delta}

print("RMSE of performance:", time_err / len(kernels))
print("RMSE of power:", power_err / len(kernels))
#print(para_dict)

p0s = [v["p0"] for v in para_dict.values()]
p_stars = [v["p_star"] for v in para_dict.values()]
gammas = [v["gamma"] for v in para_dict.values()]
cgs = [v["cg"] for v in para_dict.values()]
t0s = [v["t0"] for v in para_dict.values()]
t_stars = [v["t_star"] for v in para_dict.values()]
Ds = [v["D"] for v in para_dict.values()]
deltas = [v["delta"] for v in para_dict.values()]

p0s.sort()
p_stars.sort()
gammas.sort()
cgs.sort()
t0s.sort()
t_stars.sort()
Ds.sort()
deltas.sort()

print("p0", get_ci(p0s))
print("p_star", get_ci(p_stars))
print("gamma", get_ci(gammas))
print("cg", get_ci(cgs))
print("t0", get_ci(t0s))
print("t_star", get_ci(t_stars))
print("D", get_ci(Ds))
print("delta", get_ci(deltas))

for key in para_dict:
    print(key, para_dict[key])

with open("apps.pkl", "wb") as fp:  
    pickle.dump(para_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)   

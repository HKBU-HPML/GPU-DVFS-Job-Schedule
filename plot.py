import pandas as pd
import numpy as np
import sys,os
import random
# from sklearn import cross_validation
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr 
#import seaborn as sns
#from matplotlib_colorbar.colorbar import Colorbar

MARKERS = ['^', '<', 'o', 's', '+', '']
HATCHES = ['//', '--', '\\\\', '||', '++', '--', '..', '++', '\\\\']
GRAYS = ['#2F4F4F', '#808080', '#A9A9A9', '#778899', '#DCDCDC', '#556677', '#1D3E3E', '#808080', '#DCDCDC']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

OUTPUT_PATH = 'figures'

off_dict = {"edf+spt":"EDF-SPT", "edf+bf": "EDF-BF", "edf+wf":"EDF+WF", "lpt+ff":"LPT-FF"}

def plot_curve(df, save_filename=None):

    fig, ax = plt.subplots(figsize = (8, 6))

    algos = ["edf+bf-off-1.0", "edf+wf-off-1.0", "edf+spt-off-1.0", "lpt+ff-off-1.0", "edf+bf-on-1.0", "edf+wf-on-1.0", "edf+spt-on-1.0", "lpt+ff-on-1.0"]
    #algos = ["edf+spt-on-1.0", "edf+spt-on-0.95", "edf+spt-on-0.9", "edf+spt-on-0.85", "edf+spt-on-0.8"]
    df = df[(df.gpus_per_node == 1)]

    x_axis = np.arange(8)
    key = "total"
    for idx, algo in enumerate(algos):
        al, dvfs, theta = algo.split("-")
        if dvfs == 'on':
            ds = 1
        else:
            ds = 0
        theta = float(theta)
        tmp_df = df[(df.algo == al) & (df.dvfs_on == ds) & (df.theta == theta)].sort_values(by = ['gpus_per_node'])
        avers = tmp_df["aver_%s_E" % key].to_numpy() 
        mins = tmp_df["min_%s_E" % key].to_numpy() 
        maxs = tmp_df["max_%s_E" % key].to_numpy() 
        #x_axis = list(tmp_df["util"])
        print(algo)
        print(avers)
        #print(mins)
        #print(maxs)
        label = off_dict[al]
        if dvfs == 'on':
            label += " DVFS"
        ax.plot(x_axis, avers, linewidth = 1.5, color = COLORS[idx // 4], marker = MARKERS[idx % 4], markersize = 12, markeredgecolor = 'k', label = label, markerfacecolor='none')
        ax.fill_between(x_axis, mins, maxs, color=COLORS[idx], alpha=.1)

    fsize = 22
    ax.set_ylabel("Normalized energy consumption", size = fsize)
    #ax.set_ylim(top = 1.5, bottom = 0.95)
    ax.yaxis.set_tick_params(labelsize=fsize)
    #ax.ticklabel_format(style='sci', scilimits=(1,1), axis='y')
    ax.yaxis.offsetText.set_fontsize(fsize)

    ax.set_xlabel("Task Set Utilization", size = fsize)
    #ax.set_xlim(min(x_axis) - 100, max(x_axis) + 100)
    ax.xaxis.set_tick_params(labelsize=fsize)

    ax.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    ax.legend(fontsize=fsize, loc='upper left')

    if not save_filename:
        plt.show()
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_normalize(df, save_filename=None):

    fig, ax = plt.subplots(figsize = (8, 6))

    algos = ["edf+bf", "edf+wf", "edf+spt", "lpt+ff"]
    base_Es = df[(df.algo == "edf+spt") & (df.gpus_per_node == 1) & (df.util <= 1.6) & (df.dvfs_on == 0)].sort_values(by = ['util'])
    df = df[(df.gpus_per_node == 4) & (df.util <= 1.6) & (df.dvfs_on == 0)]

    x_axis = [0.2*(i+1) for i in range(8)]
    key = "total"
    for idx, algo in enumerate(algos):
        tmp_df = df[df.algo == algo].sort_values(by = ['util'])
        avers = tmp_df["aver_%s_E" % key].to_numpy() / base_Es["aver_%s_E" % key].to_numpy()
        mins = tmp_df["min_%s_E" % key].to_numpy() / base_Es["min_%s_E" % key].to_numpy()
        maxs = tmp_df["max_%s_E" % key].to_numpy() / base_Es["max_%s_E" % key].to_numpy()
        #x_axis = list(tmp_df["util"])
        print(algo)
        print(avers)
        #print(mins)
        #print(maxs)
        ax.plot(x_axis, avers, linewidth = 1.5, color = COLORS[idx], marker = MARKERS[idx], markersize = 12, markeredgecolor = 'k', label = algo, markerfacecolor='none')
        ax.fill_between(x_axis, mins, maxs, color=COLORS[idx], alpha=.1)

    fsize = 22
    ax.set_ylabel("Normalized energy consumption", size = fsize)
    #ax.set_ylim(top = 1.5, bottom = 0.95)
    ax.yaxis.set_tick_params(labelsize=fsize)
    #ax.ticklabel_format(style='sci', scilimits=(1,1), axis='y')
    ax.yaxis.offsetText.set_fontsize(fsize)

    ax.set_xlabel("Task Set Utilization", size = fsize)
    #ax.set_xlim(min(x_axis) - 100, max(x_axis) + 100)
    ax.xaxis.set_tick_params(labelsize=fsize)

    ax.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    ax.legend(fontsize=fsize, loc='upper left')

    if not save_filename:
        plt.show()
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

if __name__ == '__main__':
    #df = pd.read_csv("csvs/online.csv", header = 0)
    df = pd.read_csv("csvs/offline.csv", header = 0)
    #plot_normalize(df)
    plot_curve(df)

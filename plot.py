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

MARKERS = ['^', '<', 'o', 's', '+']
HATCHES = ['//', '--', '\\\\', '||', '++', '--', '..', '++', '\\\\']
GRAYS = ['#2F4F4F', '#808080', '#A9A9A9', '#778899', '#DCDCDC', '#556677', '#1D3E3E', '#808080', '#DCDCDC']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

OUTPUT_PATH = 'figures'

def plot_normalize(df, save_filename=None):

    fig, ax = plt.subplots(figsize = (8, 6))

    algos = ["edf+bf", "edf+wf", "edf+spt"]
    base_Es = df[(df.algo == "edf+spt") & (df.gpus_per_node == 1) & (df.dvfs_on == 0) & (df.util <= 0.8)].sort_values(by = ['util'])
    df = df[(df.gpus_per_node == 2) & (df.dvfs_on == 0) & (df.util <= 0.8)]

    x_axis = [0.2, 0.4, 0.6, 0.8]
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

    ax.set_ylabel("Cycles", size = 24)
    #ax.set_ylim(top = 2.4, bottom = 0.95)
    ax.yaxis.set_tick_params(labelsize=24)
    #ax.ticklabel_format(style='sci', scilimits=(1,1), axis='y')
    ax.yaxis.offsetText.set_fontsize(20)

    ax.set_xlabel("Task Set Utilization", size = 24)
    #ax.set_xlim(min(x_axis) - 100, max(x_axis) + 100)
    ax.xaxis.set_tick_params(labelsize=24)

    ax.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    ax.legend(fontsize=18, loc='upper left')

    if not save_filename:
        plt.show()
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

if __name__ == '__main__':
    df = pd.read_csv("csvs/results.csv", header = 0)
    plot_normalize(df)

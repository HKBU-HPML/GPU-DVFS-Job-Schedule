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

MARKERS = ['^', '<', 'o', 's']
HATCHES = ['//', '--', '\\\\', '||', '++', '--', '..', '++', '\\\\']
GRAYS = ['#2F4F4F', '#808080', '#A9A9A9', '#778899', '#DCDCDC', '#556677', '#1D3E3E', '#808080', '#DCDCDC']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

in_kernels = ['BlackScholes', 'matrixMulShared', 'backpropForward', 'histogram']
#in_kernels = ['BlackScholes', 'matrixMul', 'backprop', 'convolutionSeparable']
out_kernels = ['binomialOptions', 'eigenvalues', 'scanUniformUpdate', 'stereoDisparity', 'reduction', 'matrixMulGlobal', 'cfd', 'hotspot', 'dxtc', 'backpropBackward']
# experimental test
pointer = ['convolutionTexture', 'nn', 'SobolQRNG', 'reduction', 'hotspot'] 
#pointer = []
extras = ['backpropBackward', 'binomialOptions', 'cfd', 'eigenvalues', 'gaussian', 'srad', 'dxtc', 'pathfinder', 'scanUniformUpdate', 'stereoDisparity'] 
#extras = []

highest_core = 1000
lowest_core = 500
highest_mem = 1000
lowest_mem = 500

OUTPUT_PATH = 'figures'

def plot_normalize(df, save_filename=None):

    fig, ax = plt.subplots(figsize = (8, 6))

    algos = ["edf+ff", "lpt+spt"]
    df = df[df.gpus_per_node == 2]

    tmp_df = df[df.algo == algos[0]].sort_values(by = ['util'])
    avers = list(tmp_df["aver_total_E"])
    mins = list(tmp_df["min_total_E"])
    maxs = list(tmp_df["max_total_E"])
    x_axis = list(tmp_df["util"])
    print(avers)
    print(mins)
    print(maxs)
    ax.plot(x_axis, avers, linewidth = 1.5, color = COLORS[1], marker = MARKERS[1], markersize = 12, markeredgecolor = 'k', label = 'edf+ff', markerfacecolor='none')
    ax.fill_between(x_axis, mins, maxs, color='g', alpha=.1)

    tmp_df = df[df.algo == algos[1]].sort_values(by = ['util'])
    avers = list(tmp_df["aver_total_E"])
    mins = list(tmp_df["min_total_E"])
    maxs = list(tmp_df["max_total_E"])
    x_axis = list(tmp_df["util"])
    print(avers)
    print(mins)
    print(maxs)
    ax.plot(x_axis, avers, linewidth = 1.5, color = COLORS[2], marker = MARKERS[2], markersize = 12, markeredgecolor = 'k', label = 'lpt+spt', markerfacecolor='none')
    ax.fill_between(x_axis, mins, maxs, color='r', alpha=.1)

    ax.set_ylabel("Cycles", size = 24)
    ymax = ax.get_ylim()[1] * 1.35
    ymin = ax.get_ylim()[0] * 0.65
    ax.set_ylim(top = ymax, bottom = ymin)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.ticklabel_format(style='sci', scilimits=(1,1), axis='y')
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

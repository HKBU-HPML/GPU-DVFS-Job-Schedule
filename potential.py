import pandas as pd
import numpy as np
import pickle
from cluster import GPU_NAME, CORE_BASE, MEM_BASE

df = pd.read_csv("csvs/%s-dvfs-real-Performance-Power.csv" % GPU_NAME, header = 0)

def main():
    kernels = df['appName'].drop_duplicates()
    kernels.sort_values(inplace=True)
    
    saves = []
    for kernel in kernels:
    
        tmp_df = df[df.appName == kernel]
        erg = tmp_df["time/ms"] * tmp_df["power/W"]
        idx_min = np.argmin(erg)
        min_fc = tmp_df.coreF.iloc[idx_min]
        min_fm = tmp_df.memF.iloc[idx_min]
        min_pow = tmp_df['power/W'].iloc[idx_min]
        min_time = tmp_df['time/ms'].iloc[idx_min]
        min_e = erg.iloc[idx_min]
        base_info = tmp_df[(tmp_df.coreF == CORE_BASE) & (tmp_df.memF == MEM_BASE)]
        base_pow = base_info['power/W'].iloc[0]
        base_time = base_info['time/ms'].iloc[0]
        base_e = base_pow * base_time
        print(kernel, min_e, base_e, 1 - (min_e / base_e), min_fc, min_fm)
        saves.append(1 - (min_e / base_e))

    print(np.mean(saves), np.sort(saves))

if __name__ == '__main__':
    main()   

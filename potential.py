import pandas as pd
import numpy as np
import pickle

kernel_config = 'real'
gpu_config = 'gtx2070s-dvfs'
CORE_BASE = 1880
MEM_BASE = 6300
df = pd.read_csv("csvs/%s-%s-Performance-Power.csv" % (gpu_config, kernel_config), header = 0)

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
        #base_e = list(erg[(tmp_df.coreF == CORE_BASE) & (tmp_df.memF == MEM_BASE)])[0]
        print(kernel, min_e, base_e, 1 - (min_e / base_e), min_fc, min_fm)
        #print(kernel,min_fc - CORE_BASE,min_fm - MEM_BASE,int(min_time*10))
        #print(kernel,0,0,int(base_time*10))
        saves.append(1 - (min_e / base_e))

    print(np.mean(saves), np.sort(saves))

if __name__ == '__main__':
    main()   

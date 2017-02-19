import numpy as np
import pandas as pd

INPUTPATH = '/Users/anshulramachandran/Downloads/Caltech-CA-320_Actual.csv'

def get_timeseries(path):
    df = pd.read_csv(path)
    df = df.loc[df['name'] == 'mamps_actual']
    timeseries = df['value'].tolist()
    return list(map(int, timeseries))

def split_timeseries(ts):
    ts_splits = []
    curr_start = -1
    num_zeros = 0
    in_curr = False
    for i in range(len(ts)):
        if ts[i] > 0 and not in_curr:
            curr_start = i
            in_curr = True
        elif ts[i] <= 0 and in_curr:
            if ts[i-1] > 0:
                num_zeros = 0
            num_zeros += 1
            if num_zeros == 20:
                ts_splits.append(ts[curr_start:i-19])
                num_zeros = 0
                in_curr = False

    return ts_splits

ts = get_timeseries(INPUTPATH)
print(split_timeseries(ts))

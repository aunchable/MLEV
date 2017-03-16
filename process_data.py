import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy import optimize

INPUTPATH = '/Users/anshulramachandran/Downloads/Influx Data - JPL/JPL-EV-L2-57_Actual.csv'

THRESHOLD = 0.20

# Models associated with different numbers of free parameters
num_params = {
    '2' : ['lin', 'exp'],
    '4' : ['lin_lin', 'lin_exp', 'exp_lin', 'exp_exp']
}

def get_timeseries(path):
    df = pd.read_csv(path)
    df = pd.to_numeric(df.value, errors='coerce')
    df = df.dropna()
    timeseries = df.tolist()
    return list(map(int, timeseries))

def get_decay(charging_profile):

    # first filter out all values less than 1000 mA

    charging_profile = [i for i in charging_profile if i >= 1000]

    # if there are less than 10 time series values,
    # it is useless signal

    if len(charging_profile) < 10:
        return [-1]

    if int(2 * len(charging_profile)/3 % 2) ==0:
        ks = 1 + int(2 * len(charging_profile)/3)
    else:
        ks = int(2 * len(charging_profile)/3)

    clean = signal.medfilt(charging_profile, kernel_size=ks)

    mx_diff = 0
    mx_ind = 0
    for i in range(51, len(clean) - 52):
        slope1 = (clean[i+1] - clean[i-1]) / 2
        slope2 = (clean[i+50] - clean[i-50]) / 100
        if (slope2 < 0) and abs(slope1 - slope2) > mx_diff:
            mx_diff = abs(slope1 - slope2)
            mx_ind = i

    return clean[mx_ind:]

def split_timeseries(ts):
    ts_splits = []
    curr_start = -1
    num_zeros = 0
    in_curr = False
    for i in range(len(ts)):
        if ts[i] > 0 and not in_curr:
            curr_start = i
            in_curr = True
        elif ts[i] <= 100 and in_curr:
            if ts[i-1] > 0:
                num_zeros = 0
            num_zeros += 1
            if num_zeros == 2:
                if (i - curr_start > 100) and np.max(ts[curr_start:i]) > 2000:
                    dec_portion = get_decay(ts[curr_start:i])
                    if not np.array_equal(dec_portion, [-1]):
                        ts_splits.append(dec_portion)

                num_zeros = 0
                in_curr = False

    return ts_splits

############
# MODELS
############
def lin(x, y0, k):
    return [k*x_ + y0 for x_ in x]

def exp(x, y0, k):
    return [y0*np.exp(-k * x_) for x_ in x]

def lin_lin(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def lin_exp(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:y0*np.exp(-k2 * (x - x0))])

def exp_lin(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:y0*np.exp(-k1 * (x - x0)), lambda x:k2*x + y0-k2*x0])

def exp_exp(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:y0*np.exp(-k1 * (x - x0)), lambda x:y0*np.exp(-k2 * (x - x0))])


def squared_error(y1, y2):
    return np.sum([(a1 - a2)**2 for (a1, a2) in zip(y1, y2)])


def getAllFitParameters(charging_profile):

    x = np.array(range(len(charging_profile)), dtype=float)
    y = np.array(charging_profile) / max(charging_profile)
    params = {}
    errors = {}

    p, e = optimize.curve_fit(lin, x, y)
    sq = squared_error(y, lin(x, *p))
    params['lin'] = p
    errors['lin'] = sq

    p, e = optimize.curve_fit(exp, x, y)
    sq = squared_error(y, exp(x, *p))
    params['exp'] = p
    errors['exp'] = sq

    p, e = optimize.curve_fit(lin_lin, x, y)
    sq = squared_error(y, lin_lin(x, *p))
    params['lin_lin'] = p
    errors['lin_lin'] = sq

    p, e = optimize.curve_fit(exp_lin, x, y)
    sq = squared_error(y, exp_lin(x, *p))
    params['exp_lin'] = p
    errors['exp_lin'] = sq

    p, e = optimize.curve_fit(lin_exp, x, y)
    sq = squared_error(y, lin_exp(x, *p))
    params['lin_exp'] = p
    errors['lin_exp'] = sq

    p, e = optimize.curve_fit(exp_exp, x, y)
    sq = squared_error(y, exp_exp(x, *p))
    params['exp_exp'] = p
    errors['exp_exp'] = sq

    return params, errors


#TODO(anyone): make this more general
def chooseBestFit(params, errors):
    bestFits = {}
    for k, v in num_params.items():
        least_error = 1000000.
        best_fit_name = ''
        for fit in v:
            if errors[fit] < least_error:
                least_error = errors[fit]
                best_fit_name = fit
        bestFits[k] = best_fit_name

    if errors[bestFits['2']] < errors[bestFits['4']] + THRESHOLD:
        return bestFits['2'], params[bestFits['2']], errors[bestFits['2']]
    else:
        return bestFits['4'], params[bestFits['4']], errors[bestFits['4']]


ts = get_timeseries(INPUTPATH)
data = split_timeseries(ts)
print len(data)
params, errors = getAllFitParameters(data[25])
print params, errors
fit, p, e = chooseBestFit(params, errors)
print fit, p, e

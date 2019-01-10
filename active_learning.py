import random
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings("ignore")
rs = np.random.RandomState(0)
import matplotlib;matplotlib.rcParams['font.size'] = 16
import matplotlib;matplotlib.rcParams['font.family'] = 'serif'
from visualize import *
from calibration import *
from collections import defaultdict
from tqdm import tqdm
from spline import *
import collections

NUM_BINS = 10
NUM_CLASSES =  100
NUM_CLASSES_PLOT = 4
NUM_COL = 5
NUM_RUN = 50
# limit to the cases when the number of samples is less than 1000
NUM_SAMPLES = [20, 30, 40, 50, 60, 70, 80, 90, 100] + [50 * _ for _ in range(3, 19)] 
print len(NUM_SAMPLES) # should not exceed 25 because the number of subplots is set to be 25.


def active_learning(score, Y_predict, Y_true, acq_func, subset_init, candidate_list, NUM_SAMPLES, gam_ref):
    
    ece_dict = dict()
    acc_dict = dict()
    mse_dict = dict()
    subset_list = []
    
    for idx in range(len(NUM_SAMPLES)):
        if idx == 0:
            subset_list += subset_init
        else:
            n_inc = NUM_SAMPLES[idx] - NUM_SAMPLES[idx-1]
            subset_list += acq_func(n_inc, np.max(score, axis=1), candidate_list, subset_list, confi)
        ece, acc, mse, confi = spline_classification(
                                    np.max(score[subset_list], axis=1).reshape(-1, 1),
                                    np.array(Y_true == Y_predict)[subset_list] * 1,
                                    np.max(score, axis=1).reshape(-1, 1),
                                    np.array(Y_true == Y_predict) * 1,
                                    gam_ref)
        ece_dict[NUM_SAMPLES[idx]] = ece[0]
        acc_dict[NUM_SAMPLES[idx]] = acc
        mse_dict[NUM_SAMPLES[idx]] = mse
    
    return ece_dict, acc_dict, mse_dict, subset_list


def active_learning_plot(score, Y_predict, Y_true, acq_func, subset_init, candidate_list, NUM_SAMPLES, gam_ref):
    
    ece_dict = dict()
    acc_dict = dict()
    mse_dict = dict()
    subset_list = []

    fig, ax = plt.subplots(nrows=5, ncols=5)
    fig.set_figheight(9)
    fig.set_figwidth(9)
    
    for idx in range(len(NUM_SAMPLES)):
        if idx == 0:
            subset_list += subset_init
        else:
            n_inc = NUM_SAMPLES[idx] - NUM_SAMPLES[idx-1]
            subset_list += acq_func(n_inc, np.max(score, axis=1), candidate_list, subset_list, confi)
        ece, acc, mse, ax[idx / NUM_COL, idx % NUM_COL], confi = \
                                    spline_classification_plot(ax[idx / NUM_COL, idx % NUM_COL],
                                    np.max(score[subset_list], axis=1).reshape(-1, 1),
                                    np.array(Y_true == Y_predict)[subset_list] * 1,
                                    np.max(score, axis=1).reshape(-1, 1),
                                    np.array(Y_true == Y_predict) * 1,
                                    gam_ref)
        ax[idx / NUM_COL, idx % NUM_COL].set_xlabel("N=%d" % NUM_SAMPLES[idx])
        ece_dict[NUM_SAMPLES[idx]] = ece[0]
        acc_dict[NUM_SAMPLES[idx]] = acc
        mse_dict[NUM_SAMPLES[idx]] = mse
    fig.tight_layout()
    
    return ece_dict, acc_dict, mse_dict, subset_list


### different acquisition functions

def acq_random_emp(n_inc, candidate_score, candidate_list, subset_list, confi=None):
    weights = np.array([1.0 for _ in candidate_list])
    weights[subset_list] = 0
    p = weights / weights.sum()
    return np.random.choice(candidate_list, size = n_inc, replace = False, p = p).tolist()

def acq_random_unf(n_inc, candidate_score, candidate_list, subset_list, confi=None):
    num_bins = NUM_BINS
    bins = np.linspace(0, 1, num_bins+1)
    digitized = np.digitize(candidate_score, bins[1:-1])
    counter=collections.Counter(digitized.tolist())
    weights = np.array([1.0/counter[digitized[_]] for _ in candidate_list])
    weights[subset_list] = 0
    p = weights / weights.sum()
    return np.random.choice(candidate_list, size = n_inc, replace = False, p = p).tolist()


def acq_active_prb(n_inc, candidate_score, candidate_list, subset_list, confi):
    confi = sigmoid(confi) # 100 * 1
    weights = confi[:,1] - confi[:, 0]
    weights[subset_list] = 0
    p = weights/ weights.sum()
    return np.random.choice(candidate_list, size = n_inc, replace = False, p = p).tolist()

def acq_active_dtm(n_inc, candidate_score, candidate_list, subset_list, confi):
    import heapq
    confi = sigmoid(confi) # 100 * 1
    weights = confi[:,1] - confi[:, 0]
    weights[subset_list] = 0
    return heapq.nlargest(n_inc, range(len(weights)), weights.__getitem__)
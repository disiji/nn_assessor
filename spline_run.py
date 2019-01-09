import random
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")
rs = np.random.RandomState(0)
from visualize import *
from calibration import *
from collections import defaultdict
from tqdm import tqdm
from spline import *
from active_learning import *
import heapq


NUM_BINS = 10
NUM_CLASSES =  100
NUM_CLASSES_PLOT = 4
NUM_COL = 5
NUM_RUN = 100
# limit to the cases when the number of samples is less than 1000
NUM_SAMPLES = [20, 30, 40, 50, 60, 70, 80, 90, 100] + [50 * _ for _ in range(3, 19)] 
print len(NUM_SAMPLES)


DATASET = "cifar100_predictions_dropout"
data = np.genfromtxt("data/cifar100/%s.txt" % DATASET)# 10000*101
# DATASET = "svhn_predictions"
# data = np.genfromtxt("data/svhn/%s.txt" % DATASET)
# DATASET = "cifar100_adversarial_predictions"
# data = np.genfromtxt("data/cifar100_adversarial/%s.txt" % DATASET)# 10000*101


score = data[:,1:]
Y_predict = np.argmax(score, axis=1)
Y_true = data[:,0]

gam_ref = LogisticGAM(s(0, constraints='monotonic_inc')).gridsearch(
                    np.max(score, axis=1).reshape(-1, 1),
                    np.array(Y_true == Y_predict) * 1) # add a linear term


def writecsv(result_dict, filename):
    """
    result_dict: 
        a dictionary of lists. can be ece_random, ece_active, acc_random, acc_active in this case.
    """
    import csv
    with open(filename, 'w') as f:
        for key in result_dict.keys():
            f.write("%s,%s\n"%(key,",".join(["%.4f" % _ for _ in result_dict[key]])))

ece_random_emp_multi = defaultdict(list)
acc_random_emp_multi = defaultdict(list)
mse_random_emp_multi = defaultdict(list)
ece_random_unf_multi = defaultdict(list)
acc_random_unf_multi = defaultdict(list)
mse_random_unf_multi = defaultdict(list)
ece_active_prb_multi = defaultdict(list)
acc_active_prb_multi = defaultdict(list)
mse_active_prb_multi = defaultdict(list)
ece_active_dtm_multi = defaultdict(list)
acc_active_dtm_multi = defaultdict(list)
mse_active_dtm_multi = defaultdict(list)

training_list = [i for i in range(data.shape[0])]

for run_idx in tqdm(range(NUM_RUN)):
    
    subset_init = np.random.choice(training_list, size = NUM_SAMPLES[0], replace = False).tolist()
    
    ece_random_emp, acc_random_emp, mse_random_emp, subset_random_emp = active_learning(
        score, Y_predict, Y_true, acq_random_emp, subset_init, training_list, NUM_SAMPLES, gam_ref)
    ece_random_unf, acc_random_unf, mse_random_unf, subset_random_unf = active_learning(
        score, Y_predict, Y_true, acq_random_unf, subset_init, training_list, NUM_SAMPLES, gam_ref)
    ece_active_prb, acc_active_prb, mse_active_prb, subset_active_prb = active_learning(
        score, Y_predict, Y_true, acq_active_prb, subset_init, training_list, NUM_SAMPLES, gam_ref)
    ece_active_dtm, acc_active_dtm, mse_active_dtm, subset_active_dtm = active_learning(
        score, Y_predict, Y_true, acq_active_dtm, subset_init, training_list, NUM_SAMPLES, gam_ref)
    
    for _ in NUM_SAMPLES:
        ece_random_emp_multi[_].append(ece_random_emp[_])
        acc_random_emp_multi[_].append(acc_random_emp[_])
        mse_random_emp_multi[_].append(mse_random_emp[_])
        ece_random_unf_multi[_].append(ece_random_unf[_])
        acc_random_unf_multi[_].append(acc_random_unf[_])
        mse_random_unf_multi[_].append(mse_random_unf[_])
        ece_active_prb_multi[_].append(ece_active_prb[_])
        acc_active_prb_multi[_].append(acc_active_prb[_])
        mse_active_prb_multi[_].append(mse_active_prb[_])
        ece_active_dtm_multi[_].append(ece_active_dtm[_])
        acc_active_dtm_multi[_].append(acc_active_dtm[_])
        mse_active_dtm_multi[_].append(mse_active_dtm[_])
    
    print len(subset_random_emp), len(subset_random_unf), len(subset_active_prb), len(subset_active_dtm)
        
writecsv(ece_random_emp_multi, "output/%s/ece_random_emp.csv" % DATASET)
writecsv(acc_random_emp_multi, "output/%s/acc_random_emp.csv" % DATASET)
writecsv(mse_random_emp_multi, "output/%s/mse_random_emp.csv" % DATASET)
writecsv(ece_random_unf_multi, "output/%s/ece_random_unf.csv" % DATASET)
writecsv(acc_random_unf_multi, "output/%s/acc_random_unf.csv" % DATASET)
writecsv(mse_random_unf_multi, "output/%s/mse_random_unf.csv" % DATASET)
writecsv(ece_active_prb_multi, "output/%s/ece_active_prb.csv" % DATASET)
writecsv(acc_active_prb_multi, "output/%s/acc_active_prb.csv" % DATASET)
writecsv(mse_active_prb_multi, "output/%s/mse_active_prb.csv" % DATASET)
writecsv(ece_active_dtm_multi, "output/%s/ece_active_dtm.csv" % DATASET)
writecsv(acc_active_dtm_multi, "output/%s/acc_active_dtm.csv" % DATASET)
writecsv(mse_active_dtm_multi, "output/%s/mse_active_dtm.csv" % DATASET)

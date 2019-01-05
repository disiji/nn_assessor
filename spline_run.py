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
import heapq


NUM_BINS = 10
NUM_CLASSES =  100
NUM_CLASSES_PLOT = 4
NUM_COL = 5
NUM_RUN = 100
# limit to the cases when the number of samples is less than 1000
NUM_SAMPLES = [20, 50, 100] + [50 * _ for _ in range(3, 20)] 
print len(NUM_SAMPLES)


DATASET = "cifar100_predictions_dropout"
data = np.genfromtxt("data/%s.txt" % DATASET)# 10000*101
score = data[:,1:]
Y_predict = np.argmax(score, axis=1)
Y_true = data[:,0]


def writecsv(result_dict, filename):
    """
    result_dict: 
        a dictionary of lists. can be ece_random, ece_active, acc_random, acc_active in this case.
    """
    import csv
    with open(filename, 'w') as f:
        for key in result_dict.keys():
            f.write("%s,%s\n"%(key,",".join(["%.4f" % _ for _ in result_dict[key]])))

ece_random = defaultdict(list)
acc_random = defaultdict(list)
ece_active_prb = defaultdict(list)
acc_active_prb = defaultdict(list)
ece_active_dtm = defaultdict(list)
acc_active_dtm = defaultdict(list)

training_list = [i for i in range(10000)]

for run_idx in tqdm(range(NUM_RUN)):
    
    subset_init = np.random.choice(training_list, size = NUM_SAMPLES[0], replace = False).tolist()
    subset_random = []
    subset_active_prb = []
    subset_active_dtm = []
    
    for idx in range(len(NUM_SAMPLES)):
        # randomly select datapoints and feed to spline regression
        # turn off plot
        if idx == 0:
            subset_random += subset_init
        else:
            n_inc = NUM_SAMPLES[idx] - NUM_SAMPLES[idx-1]
            subset_random += np.random.choice([i for i in training_list if i not in subset_random], 
                                      size = n_inc,
                                      replace = False).tolist()
        ece, acc, confi = spline_classification(np.max(score[subset_random], axis=1).reshape(-1, 1),
                                         np.array(Y_true == Y_predict)[subset_random] * 1)
        # confi not used in random selection
        ece_random[NUM_SAMPLES[idx]].append(ece[0])
        acc_random[NUM_SAMPLES[idx]].append(acc)  
        
        
        # randomly select datapoints and feed to spline regression
        # data points are reweighted by uncertainty
        if idx == 0:
            subset_active_prb += subset_init
        else:
            n_inc = NUM_SAMPLES[idx] - NUM_SAMPLES[idx-1]
            candidate_list = [i for i in training_list if i not in subset_active_prb]
            p = weights[candidate_list,0] / weights[candidate_list].sum()
            subset_active_prb += np.random.choice(candidate_list, 
                                         size = n_inc,
                                         replace = False,
                                         p = p).tolist()
        ece, acc, confi = spline_classification(np.max(score[subset_active_prb], axis=1).reshape(-1, 1),
                                         np.array(Y_true == Y_predict)[subset_active_prb] * 1)
        confi = sigmoid(confi) # 100 * 1
        uncertainty = confi[:,1] - confi[:, 0]
        # compute probablity of each datapoint
        digitized = np.digitize(np.max(score, axis=1).reshape(-1, 1), 
                                np.linspace(0, 1, 10)) -1
        weights = uncertainty[digitized]
        ece_active_prb[NUM_SAMPLES[idx]].append(ece[0])
        acc_active_prb[NUM_SAMPLES[idx]].append(acc)
        
        # deterministically select datapoints and feed to spline regression
        if idx == 0:
            subset_active_dtm += subset_init
        else:
            n_inc = NUM_SAMPLES[idx] - NUM_SAMPLES[idx-1]
            weights[subset_active_dtm] = 0
            subset_active_dtm += heapq.nlargest(n_inc, range(len(weights)), weights.__getitem__)
        ece, acc, confi = spline_classification(np.max(score[subset_active_dtm], axis=1).reshape(-1, 1),
                                         np.array(Y_true == Y_predict)[subset_active_dtm] * 1)
        confi = sigmoid(confi) # 100 * 1
        uncertainty = confi[:,1] - confi[:, 0]
        # compute probablity of each datapoint
        digitized = np.digitize(np.max(score, axis=1).reshape(-1, 1), 
                                np.linspace(0, 1, 10)) -1
        weights = uncertainty[digitized]
        ece_active_dtm[NUM_SAMPLES[idx]].append(ece[0])
        acc_active_dtm[NUM_SAMPLES[idx]].append(acc)
        
        print len(subset_random), len(subset_active_prb), len(subset_active_dtm)
        
writecsv(ece_random, "output/%s/ece_random.csv" % DATASET)
writecsv(acc_random, "output/%s/acc_random.csv" % DATASET)
writecsv(ece_active_prb, "output/%s/ece_active_prb.csv" % DATASET)
writecsv(acc_active_prb, "output/%s/acc_active_prb.csv" % DATASET)
writecsv(ece_active_dtm, "output/%s/ece_active_dtm.csv" % DATASET)
writecsv(acc_active_dtm, "output/%s/acc_active_dtm.csv" % DATASET)

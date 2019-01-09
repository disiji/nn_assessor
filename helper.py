import numpy as np
from sklearn.preprocessing import normalize
from pygam import GAM, s, te
from pygam import LogisticGAM, s, f, l
from pygam.datasets import default
from collections import defaultdict
from sklearn.metrics import mean_squared_error

def EceEval(p, y, num_bins):
    """
    INPUT:
        p: (N, K) np.array. Each row is the normalized softmax output of from the model.
        y: (N, ) np.array. True label of data points.
        num_bins: an int. 
    OUTPUT:
        ece: expected calibration error
    """
    Y_predict = np.argmax(p, axis=1)
    Y_true = y
    bins = np.linspace(0, 1, num_bins+1)
    
    confidence = np.max(p, axis=1)
    digitized = np.digitize(confidence, bins)
    
    w = np.array([(digitized==i).sum() for i in range(num_bins)])
    w = normalize(w, norm='l1')

    confidence_bins = np.array([confidence[digitized==i].mean() for i in range(num_bins)])
    accuracy_bins = np.array([(Y_predict[digitized==i]==Y_true[digitized==i]).mean() 
                                                  for i in range(num_bins)])
    confidence_bins[np.isnan(confidence_bins)] = 0
    accuracy_bins[np.isnan(accuracy_bins)] = 0
    diff = np.absolute(confidence_bins - accuracy_bins)
    ece = np.inner(diff,w)
    return ece


def MseEval(gam1, gam2, num_bins):
    """
    INPUT:
        gam1: an GAM model
        gam2: an GAM model
        num_bins: an int. 
    OUTPUT:
        mse: mean squre error between curves fit in gam1 and gam2
    """
    XX = np.linspace(0, 1, num_bins+1)
    mse = mean_squared_error(gam1.predict_proba(XX), gam2.predict_proba(XX))
    return mse


######### load the results
def LoadCsvFromOutput(filename):
    """
    INPUT:
        filename:  a csv file. can be ece_random, ece_active, acc_random, 
                    acc_active in this case.
    OUTPUT:
        result_dict: defaultdict(list). can be ece_random, ece_active, 
                    acc_random, acc_active in this case.
    """
    import csv
    result_dict = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split(",")
            result_dict[int(line[0])].append([float(_) for _ in line[1:]])
    return result_dict
from helper import *
from pygam import GAM, s, te
from pygam import LogisticGAM, s, f, l
from pygam.datasets import default
import numpy as np

NUM_BINS = 100 # for spline 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def spline_classification_plot(ax, X, y, X_eval, y_eval, gam_ref):
    # gam = LogisticGAM(s(0)).gridsearch(X, y)
    # documentation of LogisticGAM: https://pygam.readthedocs.io/en/latest/api/logisticgam.html
    gam = LogisticGAM(s(0, constraints='monotonic_inc')).gridsearch(X, y) # add a linear term
    #XX = gam.generate_X_grid(term=0)
    XX = np.linspace(0, 1, 100)
    ax.plot(XX, gam.predict_proba(XX), c = 'g')
    ax.plot(XX, gam.confidence_intervals(XX, width=0.95), c='r', ls='--')
    # compute ece and acc after calibration
    y_ = gam.predict_proba(X_eval)
    ece = EceEval(np.array([1-y_, y_]).T , y_eval, num_bins = 100)
    mse = MseEval(gam, gam_ref, num_bins = 100)
    y_predict = y_ > 0.5
    acc = (y_predict == y_eval).mean()
    ax.text(0.05, 0.80, 'ECE=%.4f\nACC=%.4f\nMSE=%.4f'% (ece, acc, mse), size=6, ha='left', va='center',
            bbox={'facecolor':'green', 'alpha':0.5, 'pad':4})
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
#     print gam.accuracy(X, y)
#     print gam.summary()
    # compute the confidence on datapoints of X_eval
    confi = gam.confidence_intervals(X_eval, width=0.95)
    return ece, acc, mse, ax, confi

def spline_classification(X, y, X_eval, y_eval, gam_ref):
    # gam = LogisticGAM(s(0)).gridsearch(X, y)
    # documentation of LogisticGAM: https://pygam.readthedocs.io/en/latest/api/logisticgam.html
    gam = LogisticGAM(s(0, constraints='monotonic_inc')).gridsearch(X, y) # add a linear term
    #XX = gam.generate_X_grid(term=0)
    # compute ece and acc after calibration
    y_ = gam.predict_proba(X_eval)
    ece = EceEval(np.array([1-y_, y_]).T , y_eval, num_bins = 100)
    mse = MseEval(gam, gam_ref, num_bins = 100)
    y_predict = y_ > 0.5
    acc = (y_predict == y_eval).mean()
    # compute the confidence on datapoints of X_eval
    confi = gam.confidence_intervals(X_eval, width=0.95)
    return ece, acc, mse, confi

# def get_spline_uncertainty(X, y):
#     # OUTPUT:
#     #     confi: dimensionality
#     # gam = LogisticGAM(s(0)).gridsearch(X, y)
#     # documentation of LogisticGAM: https://pygam.readthedocs.io/en/latest/api/logisticgam.html
#     gam = LogisticGAM(s(0, constraints='monotonic_inc')).gridsearch(X, y) # add a linear term
#     XX = np.linspace(0, 1, NUM_BINS+1)
#     pdep, confi = gam.partial_dependence(term=0, width=.95)
#     return confi

def spline_calibration(X, y):
    gam = LogisticGAM(s(0, constraints='monotonic_inc')).gridsearch(X, y) # add a linear term
    # documentation of LogisticGAM: https://pygam.readthedocs.io/en/latest/api/logisticgam.html
    # gam = LogisticGAM(s(0, constraints='monotonic_inc')).gridsearch(X, y) # add a linear term
    # compute ece and acc after calibration
    y_ = gam.predict_proba(X)
    return y_
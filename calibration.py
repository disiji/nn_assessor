import numpy as np
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize

def isotonic_calibrate(p_val, y_val, p_test):
    num_classes = p_val.shape[1]
    # learn a list of K isotonic regressors, each for one class
    irs = [None] * num_classes
    for K in range(num_classes):
        ir = IsotonicRegression()
        irs[K] = ir.fit(p_val[:,K], np.array((y_val == K)) * 1)
    # use the irs to calibrate the oupout p_test
    p_calibrated = np.empty_like(p_test)
    for K in range(num_classes):
        p_calibrated[:, K] = irs[K].transform(p_test[:,K]).T
    p_calibrated[np.isnan(p_calibrated)] = 0
    p_calibrated = normalize(p_calibrated, axis=1, norm='l1')
    return p_calibrated
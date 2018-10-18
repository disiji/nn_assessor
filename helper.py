import numpy as np
from sklearn.preprocessing import normalize

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
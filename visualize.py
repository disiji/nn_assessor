import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize


def isotonic_regression(ax, x, y, w=[]):
    """
    INPUT:
        ax: an Axes object
        x: (N, ) np.array
        y: (N, ) np.array
        w: None or a list of length N.
    OUTPUT:
        ax: an Axes object
    """
    
    if len(w) == 0:
        w = [1.0 for _ in y]
    n = len(y)
        
    # Fit IsotonicRegression and LinearRegression models
    ir = IsotonicRegression()
    y_ = ir.fit_transform(x, y, sample_weight = w)
    
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y, sample_weight = w)  # x needs to be 2d for LinearRegression
    
    # Plot result
    segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
    lc = LineCollection(segments, zorder=0)
    lc.set_array(np.ones(len(y)))
    lc.set_linewidths(np.full(n, 0.5))

    ax.plot(x, y, 'r.', markersize=12, alpha = 0.5)
    ax.plot(x, y_, 'g^', markersize=12, alpha = 0.5)
    ax.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
    #ax.gca().add_collection(lc)
    #ax.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
    
    return ax

def reliability_plot(ax, p, y, num_bins=10):
    """
    INPUT:
        ax: an Axes object
        p: (N, K) np.array. Each row is the normalized softmax output of from the model.
        y: (N, ) np.array. True label of data points.
        num_bins: an int. 
    OUTPUT:
        ece: 
        acc: 
        ax: an Axes object
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
    acc = (Y_predict == Y_true).mean()
    
    ax.grid(True)
    ax.scatter([i+0.5 for i in range(num_bins)], accuracy_bins,label="Accuracy",marker="^",s=100)
    ax.plot(np.linspace(0, 1, 11),linestyle="--",linewidth=3,c = "gray")
    ax.text(0.5, 0.9, 'ECE=%.4f\nACC=%.4f'% (ece, acc), size=14, ha='left', va='center',
            bbox={'facecolor':'green', 'alpha':0.5, 'pad':4})
    ax.set_ylim((0.0,1.0))
    ax.set_xlim((0.0,num_bins))
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    
#     ax.bar(range(NUM_BINS),confidence_bins,color='r',width=1.0,label="Confidence",alpha=0.6)
#     ax.bar(range(NUM_BINS),accuracy_bins,color='b',width=1.0,label="Accuracy",alpha=0.8)
    ax.set_xticks(range(0, 1+num_bins, 2))
    ax.set_xticklabels(["%.1f" % i for  i in bins][::2])
    return ece, acc, ax
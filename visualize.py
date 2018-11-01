import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from helper import EceEval


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

    ax.plot(x, y, 'r.', markersize=12, alpha = 0.2)
    ax.plot(x, y_, 'g^', markersize=12, alpha = 0.2)
    ax.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # compute ece and acc after calibration
    ece = EceEval(np.array([1-y_, y_]).T , y, num_bins = 20)
    y_predict = y_ > 0.5
    acc = (y_predict == y).mean()
    
    ax.text(0.05, 0.8, 'ECE=%.4f\nACC=%.4f'% (ece, acc), size=14, ha='left', va='center',
            bbox={'facecolor':'green', 'alpha':0.5, 'pad':4})
    
    return ax

def gp_regression(ax, x, y):
    """
    INPUT:
        ax: an Axes object
        x: (N, ) np.array
        y: (N, ) np.array
    OUTPUT:
        ax: an Axes object
    """        
    # Fit GaussianProcessRegressor and LinearRegression models
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(x[:, np.newaxis], y)
    y_, sigma_ = gp.predict(x[:, np.newaxis], return_std=True)    
    print y_.shape
    
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression
    

    ax.plot(x, y, 'r.', markersize=12, alpha = 0.2)
    ax.plot(x, y_, 'b-', label=u'Prediction')
    
    x_plot = np.atleast_2d(np.linspace(0, 1, 100)).T
    y_plot, sigma = gp.predict(x_plot, return_std=True)
    ax.fill(np.concatenate([x_plot, x_plot[::-1]]),
             np.concatenate([y_plot - 1.9600 * sigma,
                            (y_plot + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None')
    
    ax.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # compute ece and acc after calibration
    ece = EceEval(np.array([1-y_, y_]).T , y, num_bins = 20)
    y_predict = y_ > 0.5
    acc = (y_predict == y).mean()
    
    ax.text(0.05, 0.8, 'ECE=%.4f\nACC=%.4f'% (ece, acc), size=14, ha='left', va='center',
            bbox={'facecolor':'green', 'alpha':0.5, 'pad':4})
    
    return ax

def gpc_sklearn(ax, x, y, kernel):
    """
    Implemented with GaussianProcessClassifier in sklearn.gaussisan_process.
    The implementation is based on Algorithm 3.1, 3.2, and 5.1 of GPML. 
    The Laplace approximation is used for approximating the non-Gaussian posterior by a Gaussian.
    The implementation is restricted to using the logistic link function.
    
    INPUT:
        ax: an Axes object
        x: (N, ) np.array
        y: (N, ) np.array
        kernel: sklearn.gaussian_process.kernels object. Used to initialize GaussianProcessClassifier
    OUTPUT:
        ax: an Axes object
    """
    # Fit GaussianProcessClassification and LinearRegression models
    gpc = GaussianProcessClassifier(kernel=kernel)
    gpc.fit(x[:, np.newaxis], y)
    print("\nLearned kernel: %s" % gpc.kernel_)
    y_ = gpc.predict_proba(x[:, np.newaxis])[:,1]
    
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression
    
    # Plot 
    ax.plot(x, y, 'r.', markersize=12, alpha = 0.2)
    ax.plot(x, y_, 'b^', markersize=12, alpha = 0.2)
    
    ax.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 
    
    # compute ece and acc after calibration
    ece = EceEval(np.array([1-y_, y_]).T , y, num_bins = 20)
    y_predict = y_ > 0.5
    acc = (y_predict == y).mean()
    
    ax.text(0.05, 0.8, 'ECE=%.4f\nACC=%.4f'% (ece, acc), size=14, ha='left', va='center',
            bbox={'facecolor':'green', 'alpha':0.5, 'pad':4})
    
    return ax


def reliability_plot(ax, p, y, num_bins=10):
    """
    INPUT:
        ax: an Axes object
        p: (N, K) np.array. Each row is the normalized softmax output of from the model.
        y: (N, ) np.array. True label of data points.
        num_bins: an int. 
    OUTPUT:
        ece: expected calibration error
        acc: accuracy
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
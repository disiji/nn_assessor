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
import GPy


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
    ax.plot(x, y_, 'b^', markersize=12, alpha = 0.2)
    
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
    ece = EceEval(np.array([1-y_, y_]).T , y, num_bins = 100)
    y_predict = y_ > 0.5
    acc = (y_predict == y).mean()
    
    ax.text(0.05, 0.8, 'ECE=%.4f\nACC=%.4f'% (ece, acc), size=14, ha='left', va='center',
            bbox={'facecolor':'green', 'alpha':0.5, 'pad':4})
    
    return ax

def gpc_sklearn(ax, x, y, kernel, optimizer="fmin_l_bfgs_b"):
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
        optimizer : 
            string or callable.
            Can either be one of the internally supported optimizers for optimizing the kernel's parameters,
            specified by a string, or an externally defined optimizer passed as a callable.
            If a callable is passed, it must have the signature.
            If None is passed, the kernel's parameters are kept [
        ax: an Axes object
    """
    # Fit GaussianProcessClassification and LinearRegression models
    gpc = GaussianProcessClassifier(kernel=kernel, optimizer=optimizer)
    gpc.fit(x[:, np.newaxis], y)
    print("\nLearned kernel: %s" % gpc.kernel_)
    y_ = gpc.predict_proba(x[:, np.newaxis])[:,1]
    
    xs = np.linspace(np.min(x), np.max(x), 1000)
    ys = gpc.predict_proba(xs[:, np.newaxis])[:,1]
    
    # lr = LinearRegression()
    # lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression
    
    # Plot 
    # ax.plot(x, y, 'r.', markersize=12, alpha = 0.2)
    ax.plot(xs, ys, markersize=12, alpha = 0.2)
    
    # ax.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
    # ax.set_xlim(-0.1, 1.1)
    # ax.set_ylim(-0.1, 1.1)
    
    # compute ece and acc after calibration
    ece = EceEval(np.array([1-y_, y_]).T , y, num_bins = 100)
    y_predict = y_ > 0.5
    acc = (y_predict == y).mean()
    
    ax.text(0.05, 0.8, 'ECE=%.4f\nACC=%.4f'% (ece, acc), size=14, ha='left', va='center',
            bbox={'facecolor':'green', 'alpha':0.5, 'pad':4})
    
    return ax

def gpc_gpy(ax, x, y, kernel=None):
    """
    Implemented with GPy. The latent function values modeled by GP are squashed through the probit function. 
    We interleave runs of EP with optimization of the parameters using gradient descent methods. 
    EP is a method for fitting a Gaussian to the posterior, p(f|y) of the latent (hidden) function, given the data. 
    Whilst the parameters are being optimized, the EP approximation (the parameters of the EP factors) is fixed.
    
    INPUT:
        ax: an Axes object
        x: (N, ) np.array
        y: (N, ) np.array
        kernel: A GPy.kern object or None.
    OUTPUT:
        ax: an Axes object
    """
    # Fit GaussianProcessClassification and LinearRegression models
    if kernel == None:
        m = GPy.models.GPClassification(X=x[:, np.newaxis],Y=y[:, np.newaxis])
    else:
        m = GPy.core.GP(
            X=x[:, np.newaxis],
            Y=y[:, np.newaxis],
            kernel=kernel, 
            inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
            likelihood=GPy.likelihoods.Bernoulli()
        )
    print m, '\n'
    for i in range(3):
        m.optimize('bfgs', max_iters=100) #first runs EP and then optimizes the kernel parameters
        print 'iteration:', i,
        print m
        print ""
    y_ = m.predict(x[:, np.newaxis])[0][:,0]
    
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression
    
    # Plot data, p(y=1|x) in GPC, linear regression
    #m.plot()
    ax.plot(x, y, 'r.', markersize=12, alpha = 0.2)
    ax.plot(x, y_, 'b^', markersize=12, alpha = 0.2)
    ax.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    
    # Plot 
    
    # compute ece and acc after calibration
    ece = EceEval(np.array([1-y_, y_]).T , y, num_bins = 100)
    y_predict = y_ > 0.5
    acc = (y_predict == y).mean()
    
    ax.text(0.05, 0.70, 'ECE=%.4f\nACC=%.4f'% (ece, acc), size=14, ha='left', va='center',
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
    digitized = np.digitize(confidence, bins[1:-1])
    
    w = np.array([(digitized==i).sum() for i in range(1, num_bins+1)])
    w = normalize(w, norm='l1')

    confidence_bins = np.array([confidence[digitized==i].mean() for i in range(1, num_bins+1)])
    accuracy_bins = np.array([(Y_predict[digitized==i]==Y_true[digitized==i]).mean() 
                                                  for i in range(1,num_bins+1)])
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

def reliability_plot_binary(ax, p, y, num_bins=10):
    Y_predict = p > 0.5
    Y_true = y
    bins = np.linspace(0, 1, num_bins+1)
    digitized = np.digitize(p, bins[1:-1])
    
    w = np.array([(digitized==i).sum() for i in range(1, num_bins+1)])
    w = normalize(w, norm='l1')

    p_bins = np.array([p[digitized==i].mean() for i in range(1, num_bins+1)])
    accuracy_bins = np.array([(Y_true[digitized==i]==1.0).mean() 
                                                  for i in range(1, num_bins+1)])
    #p_bins[np.isnan(p_bins)] = 0
    #accuracy_bins[np.isnan(p_bins)] = 0
    #diff = np.absolute(p_bins - accuracy_bins)
    #ece = np.inner(diff,w)
    #acc = (Y_predict == Y_true).mean()
    ece = None
    acc = None
    ax.grid(True)
    
    ax.scatter([i+0.05 for i in range(num_bins)], accuracy_bins,label="Accuracy",marker="^",s=100)
    
    ax.plot(np.linspace(0, 1, 11),linestyle="--",linewidth=3,c = "gray")
    #ax.text(0.5, 0.9, 'ECE=%.4f\nACC=%.4f'% (ece, acc), size=14, ha='left', va='center',
    #bbox={'facecolor':'green', 'alpha':0.5, 'pad':4})
#     ax.set_ylim((0.0,1.0))
#     ax.set_xlim((0.0,num_bins))
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(0, 1+num_bins, 2))
    ax.set_xticklabels(["%.1f" % i for  i in bins][::2])
    return ece, acc, ax


# plot Brightness v.s. accuracy
def brightness_plot_binary(ax, brightness, Y_predicted, Y_true, num_bins=10):
    bins = np.linspace(brightness.min(), brightness.max(), num_bins+1)
    digitized = np.digitize(brightness, bins[1:-1])
    brightness_bins = np.array([brightness[digitized==i].mean() for i in range(1, num_bins+1)])
    accuracy_bins = np.array([(Y_true[digitized==i]==Y_predicted[digitized==i]).mean() for i in range(1, num_bins+1)])
    
    ax.grid(True)
    ax.scatter(brightness_bins, accuracy_bins,label="Accuracy", marker="*",s=12)
    ax.set_xticklabels(["%.1f" % i for  i in bins][::2])
    return ax


def plot_metric_single_run(ece_random_emp, mce_random_emp, brier_random_emp, acc_random_emp, mse_random_emp,\
                          ece_random_unf, mce_random_unf, brier_random_unf, acc_random_unf, mse_random_unf,\
                          ece_random_ent, mce_random_ent, brier_random_ent, acc_random_ent, mse_random_ent,\
                          ece_active_prb, mce_active_prb, brier_active_prb, acc_active_prb, mse_active_prb,\
                          ece_active_dtm, mce_active_dtm, brier_active_dtm, acc_active_dtm, mse_active_dtm, NUM_SAMPLES):
    fig, ax = plt.subplots(nrows=1, ncols=5)
    fig.set_figheight(3)
    fig.set_figwidth(16)
    ax[0].plot(NUM_SAMPLES, 
                 [ece_active_prb[_] for _ in NUM_SAMPLES], 
                 c = 'r', label="Active_Prb")
    ax[0].plot(NUM_SAMPLES, 
                 [ece_active_dtm[_] for _ in NUM_SAMPLES], 
                 c = 'g', label="Active_Dtm")
    ax[0].plot(NUM_SAMPLES, 
                 [ece_random_emp[_] for _ in NUM_SAMPLES], 
                 c = 'b', label="Random_Emp")
    ax[0].plot(NUM_SAMPLES, 
                 [ece_random_unf[_] for _ in NUM_SAMPLES], 
                 c = 'c', label="Random_Unf")
    ax[0].plot(NUM_SAMPLES, 
                 [ece_random_ent[_] for _ in NUM_SAMPLES], 
                 c = 'm', label="Random_Ent")
    ax[0].set_xlabel("#datapoints")
    ax[0].set_title("ECE")
    ax[0].legend()
    ax[1].plot(NUM_SAMPLES, 
                 [acc_active_prb[_] for _ in NUM_SAMPLES], 
                 c = 'r', label="Active_Prb")
    ax[1].plot(NUM_SAMPLES, 
                 [acc_active_dtm[_] for _ in NUM_SAMPLES], 
                 c = 'g', label="Active_Dtm")
    ax[1].plot(NUM_SAMPLES, 
                 [acc_random_emp[_] for _ in NUM_SAMPLES],
                 c = 'b', label="Random_Emp")
    ax[1].plot(NUM_SAMPLES, 
                 [acc_random_unf[_] for _ in NUM_SAMPLES],
                 c = 'c', label="Random_Unf")
    ax[1].plot(NUM_SAMPLES, 
                 [acc_random_ent[_] for _ in NUM_SAMPLES],
                 c = 'm', label="Random_Ent")
    ax[1].set_xlabel("#datapoints")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[2].plot(NUM_SAMPLES, 
                 [mse_active_prb[_] for _ in NUM_SAMPLES], 
                 c = 'r', label="Active_Prb")
    ax[2].plot(NUM_SAMPLES, 
                 [mse_active_dtm[_] for _ in NUM_SAMPLES], 
                 c = 'g', label="Active_Dtm")
    ax[2].plot(NUM_SAMPLES, 
                 [mse_random_emp[_] for _ in NUM_SAMPLES],
                 c = 'b', label="Random_Emp")
    ax[2].plot(NUM_SAMPLES, 
                 [mse_random_unf[_] for _ in NUM_SAMPLES],
                 c = 'c', label="Random_Unf")
    ax[2].plot(NUM_SAMPLES, 
                 [mse_random_ent[_] for _ in NUM_SAMPLES],
                 c = 'm', label="Random_Ent")
    ax[2].set_xlabel("#datapoints")
    ax[2].set_title("MSE from GAM_ref")
    ax[2].legend()
    ax[3].plot(NUM_SAMPLES, 
                 [mce_active_prb[_] for _ in NUM_SAMPLES], 
                 c = 'r', label="Active_Prb")
    ax[3].plot(NUM_SAMPLES, 
                 [mce_active_dtm[_] for _ in NUM_SAMPLES], 
                 c = 'g', label="Active_Dtm")
    ax[3].plot(NUM_SAMPLES, 
                 [mce_random_emp[_] for _ in NUM_SAMPLES],
                 c = 'b', label="Random_Emp")
    ax[3].plot(NUM_SAMPLES, 
                 [mce_random_unf[_] for _ in NUM_SAMPLES],
                 c = 'c', label="Random_Unf")
    ax[3].plot(NUM_SAMPLES, 
                 [mce_random_ent[_] for _ in NUM_SAMPLES],
                 c = 'm', label="Random_Ent")
    ax[3].set_xlabel("#datapoints")
    ax[3].set_title("MCE")
    ax[3].legend()
    ax[4].plot(NUM_SAMPLES, 
                 [brier_active_prb[_] for _ in NUM_SAMPLES], 
                 c = 'r', label="Active_Prb")
    ax[4].plot(NUM_SAMPLES, 
                 [brier_active_dtm[_] for _ in NUM_SAMPLES], 
                 c = 'g', label="Active_Dtm")
    ax[4].plot(NUM_SAMPLES, 
                 [brier_random_emp[_] for _ in NUM_SAMPLES],
                 c = 'b', label="Random_Emp")
    ax[4].plot(NUM_SAMPLES, 
                 [brier_random_unf[_] for _ in NUM_SAMPLES],
                 c = 'c', label="Random_Unf")
    ax[4].plot(NUM_SAMPLES, 
                 [brier_random_ent[_] for _ in NUM_SAMPLES],
                 c = 'm', label="Random_Ent")
    ax[4].set_xlabel("#datapoints")
    ax[4].set_title("Brier Score")
    ax[4].legend()
    plt.rc('legend',**{'fontsize':6})
    fig.tight_layout()
    plt.show()


def plot_metric_multi_run(
    ece_random_emp_multi_run, mce_random_emp_multi_run, brier_random_emp_multi_run, acc_random_emp_multi_run, mse_random_emp_multi_run,\
    ece_random_unf_multi_run, mce_random_unf_multi_run, brier_random_unf_multi_run, acc_random_unf_multi_run, mse_random_unf_multi_run,\
    ece_random_ent_multi_run, mce_random_ent_multi_run, brier_random_ent_multi_run, acc_random_ent_multi_run, mse_random_ent_multi_run,\
    ece_active_prb_multi_run, mce_active_prb_multi_run, brier_active_prb_multi_run, acc_active_prb_multi_run, mse_active_prb_multi_run,\
    ece_active_dtm_multi_run, mce_active_dtm_multi_run, brier_active_dtm_multi_run, acc_active_dtm_multi_run, mse_active_dtm_multi_run,
    NUM_SAMPLES):
    
    fig, ax = plt.subplots(nrows=1, ncols=5)
    fig.set_figheight(3)
    fig.set_figwidth(16)
    ax[0].errorbar(NUM_SAMPLES, 
                 [np.mean(ece_active_prb_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(ece_active_prb_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'r', label="Active_Prb")
    ax[0].errorbar(NUM_SAMPLES, 
                 [np.mean(ece_active_dtm_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(ece_active_dtm_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'g', label="Active_Dtm")
    ax[0].errorbar(NUM_SAMPLES, 
                 [np.mean(ece_random_emp_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(ece_random_emp_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'b', label="Random_Emp")
    ax[0].errorbar(NUM_SAMPLES, 
                 [np.mean(ece_random_unf_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(ece_random_unf_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'c', label="Random_Unf")
    ax[0].errorbar(NUM_SAMPLES, 
                 [np.mean(ece_random_ent_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(ece_random_ent_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'm', label="Random_Ent")
    ax[0].set_xlabel("#datapoints")
    ax[0].set_title("ECE")
    ax[0].legend()
    ax[1].errorbar(NUM_SAMPLES, 
                 [np.mean(acc_active_prb_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(acc_active_prb_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'r', label="Active_Prb")
    ax[1].errorbar(NUM_SAMPLES, 
                 [np.mean(acc_active_dtm_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(acc_active_dtm_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'g', label="Active_Dtm")
    ax[1].errorbar(NUM_SAMPLES, 
                 [np.mean(acc_random_emp_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(acc_random_emp_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'b', label="Random_Emp")
    ax[1].errorbar(NUM_SAMPLES, 
                 [np.mean(acc_random_unf_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(acc_random_unf_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'c', label="Random_Unf")
    ax[1].errorbar(NUM_SAMPLES, 
                 [np.mean(acc_random_ent_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(acc_random_ent_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'm', label="Random_Ent")
    ax[1].set_xlabel("#datapoints")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[2].errorbar(NUM_SAMPLES, 
                 [np.mean(mse_active_prb_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mse_active_prb_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'r', label="Active_Prb")
    ax[2].errorbar(NUM_SAMPLES, 
                 [np.mean(mse_active_dtm_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mse_active_dtm_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'g', label="Active_Dtm")
    ax[2].errorbar(NUM_SAMPLES, 
                 [np.mean(mse_random_emp_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mse_random_emp_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'b', label="Random_Emp")
    ax[2].errorbar(NUM_SAMPLES, 
                 [np.mean(mse_random_unf_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mse_random_unf_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'c', label="Random_Unf")
    ax[2].errorbar(NUM_SAMPLES, 
                 [np.mean(mse_random_ent_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mse_random_ent_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'm', label="Random_Ent")
    ax[2].set_xlabel("#datapoints")
    ax[2].set_title("MSE from GAM_ref")
    ax[2].legend()
    ax[3].errorbar(NUM_SAMPLES, 
                 [np.mean(mce_active_prb_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mce_active_prb_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'r', label="Active_Prb")
    ax[3].errorbar(NUM_SAMPLES, 
                 [np.mean(mce_active_dtm_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mce_active_dtm_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'g', label="Active_Dtm")
    ax[3].errorbar(NUM_SAMPLES, 
                 [np.mean(mce_random_emp_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mce_random_emp_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'b', label="Random_Emp")
    ax[3].errorbar(NUM_SAMPLES, 
                 [np.mean(mce_random_unf_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mce_random_unf_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'c', label="Random_Unf")
    ax[3].errorbar(NUM_SAMPLES, 
                 [np.mean(mce_random_ent_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(mce_random_ent_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'm', label="Random_Ent")
    ax[3].set_xlabel("#datapoints")
    ax[3].set_title("MCE")
    ax[3].legend()
    ax[4].errorbar(NUM_SAMPLES, 
                 [np.mean(brier_active_prb_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(brier_active_prb_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'r', label="Active_Prb")
    ax[4].errorbar(NUM_SAMPLES, 
                 [np.mean(brier_active_dtm_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(brier_active_dtm_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'g', label="Active_Dtm")
    ax[4].errorbar(NUM_SAMPLES, 
                 [np.mean(brier_random_emp_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(brier_random_emp_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'b', label="Random_Emp")
    ax[4].errorbar(NUM_SAMPLES, 
                 [np.mean(brier_random_unf_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(brier_random_unf_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'c', label="Random_Unf")
    ax[4].errorbar(NUM_SAMPLES, 
                 [np.mean(brier_random_ent_multi_run[_]) for _ in NUM_SAMPLES], 
                 yerr = [np.std(brier_random_ent_multi_run[_]) for _ in NUM_SAMPLES],
                 c = 'm', label="Random_Ent")
    ax[4].set_xlabel("#datapoints")
    ax[4].set_title("Brier Score")
    ax[4].legend()
    plt.rc('legend',**{'fontsize':6})
    fig.tight_layout()
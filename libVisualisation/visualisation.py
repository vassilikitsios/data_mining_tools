#=====================================================================
# import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#=====================================================================
def plot_prediction(y_test, y_predict,filename):
    '''
    Plots output prediction against true output for regression studies.
    '''
    #matplotlib.rc('font', size=28, family='latex')
    matplotlib.rc('font', size=20, family='sans-serif')
    matplotlib.rcParams['xtick.major.pad']=12
    matplotlib.rcParams['ytick.major.pad']=12
    plt.figure()
    plt.xlabel('data')
    plt.ylabel('prediction')
    plt.axis('equal')
    x_min = min(np.min(y_test),np.min(y_predict))
    x_max = max(np.max(y_test),np.max(y_predict))
    plt.xlim([x_min,x_max])
    plt.ylim([x_min,x_max])
    plt.plot(y_test, y_predict, 'go', y_test, y_test, 'k-')
    plt.subplots_adjust(left=0.22, bottom=0.2, right=0.95, top=0.95)
    plt.savefig(filename)
    plt.close()
    return

#=====================================================================
def plot_error(y_test, y_predict, filename):
    '''
    Plots output prediction against squared difference between true output 
    and output prediction for regression studies.
    '''
    #matplotlib.rc('font', size=28, family='latex')
    matplotlib.rc('font', size=20, family='sans-serif')
    matplotlib.rcParams['xtick.major.pad']=12
    matplotlib.rcParams['ytick.major.pad']=12
    plt.figure()
    plt.xlabel('data')
    plt.ylabel('squred error')
    plt.plot(y_test, np.power(y_predict-y_test,2.0)/float(len(y_test)), 'o')
    plt.subplots_adjust(left=0.22, bottom=0.2, right=0.95, top=0.95)
    plt.savefig(filename)
    plt.close()
    return

#=====================================================================
def plot_performance(parameter, errors_validation, errors_train, y_label, x_label, plot_type, filename):
    '''
    General subroutine to plot performance measure of validation and training
    sets against common input parameter (eg: regularisation level).
    '''    
    #matplotlib.rc('font', size=28, family='latex')
    matplotlib.rc('font', size=20, family='sans-serif')
    matplotlib.rcParams['xtick.major.pad']=12
    matplotlib.rcParams['ytick.major.pad']=12
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if (plot_type=='logX') or (plot_type=='logXY'):
        ax.set_xscale('log')
    if (plot_type=='logY') or (plot_type=='logXY'):
        ax.set_yscale('log')
    plt.plot(parameter, errors_validation, 'ro', parameter, errors_train, 'bo')
    plt.subplots_adjust(left=0.22, bottom=0.2, right=0.95, top=0.95)
    fig.savefig(filename)
    plt.close()
    return

#=====================================================================
# EOF
#=====================================================================
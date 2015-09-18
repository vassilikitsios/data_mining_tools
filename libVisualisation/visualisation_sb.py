#=====================================================================
# import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#=====================================================================
def plot_pair(df,filename):
    '''
    Plot scatter plots of input and output variables against each other.
    '''
    plt.figure()
    sns.set()
    sns.pairplot(df,size=1.5)
    plt.savefig(filename)
    plt.close()
    return

#=====================================================================
def plot_correlations(df,include_numbers,filename):
    '''
    Plot maps of cross correlations of input and output variables.
    '''
    plt.figure()
    sns.set()
    sns.corrplot(df,annot=include_numbers)
    plt.savefig(filename)
    plt.close()
    return

#=====================================================================
# EOF
#=====================================================================
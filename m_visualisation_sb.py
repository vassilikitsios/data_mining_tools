#=====================================================================
# import libraries
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

#=====================================================================
def plot_pair(df,filename):
	plt.figure()
	sns.set()
	sns.pairplot(df,size=1.5)
	plt.savefig(filename)
	plt.close()
	return

#=====================================================================
def plot_correlations(df,include_numbers,filename):
	plt.figure()
	sns.set()
	sns.corrplot(df,annot=include_numbers)
	plt.savefig(filename)
	plt.close()
	return

#=====================================================================

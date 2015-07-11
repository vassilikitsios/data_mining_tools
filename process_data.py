# #!/usr/bin/env python
#=====================================================================
# import libraries
import sys
import pandas as pd
import numpy as np
import scipy as sci
import sklearn as skl
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

import m_clean as cl
import m_machine_learning as ml
import m_visualisation as vis
import m_io as io 
import m_theano_logistic_regression as thlr
import m_theano_neural_net as thnn

#=====================================================================
# README:
#=====================================================================
# Code used to generate results for various blog posts hosted at http://dataminingtheworld.blogspot.com.au/
# including:
#	linear regression of life expectancy with regularisation and learning curves
# 	classification of OECD countries with regularisation and learning curves

# To run code with plots output to command line
#	~/Applications/anaconda/bin/ipython qtconsole --pylab=inline

#=====================================================================
# to do list:
#=====================================================================
# 1) wrap up regression and classification subroutines into a python class
# 2) change years from string to time series 

#=====================================================================
# Main program
#=====================================================================
if __name__ == '__main__':

	print('\n===================================================')
	print('Running '.format(sys.argv[0]))
	print('===================================================\n')

	#-------------------------------------------------------------
	perform_regression = False
	#perform_regression = True

	#perform_classification = False
	perform_classification = True

	perform_visualisation = False
	#perform_visualisation = True

	#-------------------------------------------------------------
	df  = cl.read_and_clean_data()		# read data and drop redundant fields
	df2 = (df - df.mean()) / df.std()	# standardise
	df2 = df2.dropna()			# drop missing values

	#-------------------------------------------------------------
	if (perform_regression):
		X,y = cl.set_up_regression_matrices(df2)
		min_error = ml.perform_lasso_regression(X,y,"lifeExp")
		del X, y

	#-------------------------------------------------------------
	if (perform_classification) or (perform_visualisation):
		df_oecd=pd.read_csv('./0data/classifier_oecd.csv', header=0,index_col=0)
		df_oecd = df_oecd.T.ffill().T							# forward fill missing values
		df_oecd_s = pd.DataFrame(df_oecd.stack().values, columns=['oecd'], index=df_oecd.stack().index)
		df2 = pd.merge(df2,df_oecd_s,left_index=True,right_index=True,how='outer')	# add oecd classification field
		df2 = df2.unstack(0).bfill().ffill().stack()					# back & forward fill missing values
		df2 = df2.dropna()

	#-------------------------------------------------------------
	if (perform_classification):
		Xpos,Xneg,ypos,yneg = cl.set_up_classification_matrices(df2)
		#ml.perform_logistic_classification(Xpos,Xneg,ypos,yneg,"oecd","l1")
		#ml.perform_logistic_classification(Xpos,Xneg,ypos,yneg,"oecd","l2")
		#max_degree = 1			# degree of polynomial feature Kernal, degree 0 is Gaussian
		#for d in range(0,max_degree+1):
		#	ml.perform_svm_classification(Xpos,Xneg,ypos,yneg,"oecd",d)
		#thlr.perform_classification(Xpos,Xneg,ypos,yneg)
		#thnn.perform_classification(Xpos,Xneg,ypos,yneg,"oecd","l1")
		thnn.perform_classification(Xpos,Xneg,ypos,yneg,"oecd","l2")
		del Xpos, Xneg, ypos, yneg

	#-------------------------------------------------------------
	if (perform_visualisation):
		print('Visualising the results ...\n')
		df3 = df2.copy()
		df3.drop('death', axis=1, inplace=True)
		df3.drop('gdpPC', axis=1, inplace=True)
		df3.drop('healthPCGov', axis=1, inplace=True)
		df3.drop('menSchool', axis=1, inplace=True)
		df3.drop('womenSchool', axis=1, inplace=True)
		df3.drop('pop', axis=1, inplace=True)
		df3.drop('popGrowth', axis=1, inplace=True)
		df3.drop('immigration', axis=1, inplace=True)

		import m_visualisation_sb as vis_sb
		vis_sb.plot_pair(df2, 'pairplot.all.png')
		vis_sb.plot_pair(df3, 'pairplot.png')
		vis_sb.plot_correlations(df2, False, 'corrplot.all.png')
		vis_sb.plot_correlations(df3, True, 'corrplot.png')

		df2.drop('oecd', axis=1, inplace=True)
		df3.drop('oecd', axis=1, inplace=True)
		vis_sb.plot_correlations(df2, False, 'corrplot.all_no_OECD.png')
		vis_sb.plot_correlations(df3, True, 'corrplot.no_OECD.png')
		vis_sb.plot_pair(df2, 'pairplot.all_no_OECD.png')
		vis_sb.plot_pair(df3, 'pairplot.no_OECD.png')

	print ('\n===================================================')
	print ('Code complete.')
	print ('===================================================\n')

#=====================================================================
# EOF
#=====================================================================

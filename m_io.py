#=====================================================================
# import libraries
import sys
import pandas as pd
import numpy as np
import scipy as sci
import sklearn as skl

#=====================================================================
def write_regression_sensitivity_results(alphas,errors,scores,filename,input_var):
	file = open(filename,'w')
	file.write('#\n')
	file.write('# {0}, squared error, score\n'.format(input_var))
	file.write('#\n')
	for i in range(0,len(alphas)):
		file.write('%18.10e %18.10e %18.10e\n' % (alphas[i], errors[i], scores[i]))
	file.close()
	return

#=====================================================================
def write_regression_regularisation_results(alphas,errors,scores,filename):
	write_regression_sensitivity_results(alphas,errors,scores,filename,'regularisation parameter')
	return

#=====================================================================
def write_regression_learning_curve_results(alphas,errors,scores,filename):
	write_regression_sensitivity_results(alphas,errors,scores,filename,'proportion of total training samples')
	return

#=====================================================================
def write_classification_sensitivity_results(alphas,P,R,F1,scores,filename,input_var):
	file = open(filename,'w')
	file.write('#\n')
	file.write('# {0}, precision, recall, F1 score, score\n'.format(input_var))
	file.write('#\n')
	for i in range(0,len(alphas)):
		file.write('%18.10e %18.10e %18.10e %18.10e %18.10e\n' % (alphas[i], P[i], R[i], F1[i], scores[i]))
	file.close()
	return

#=====================================================================
def write_classification_regularisation_results(alphas,P,R,F1,scores,filename):
	write_classification_sensitivity_results(alphas,P,R,F1,scores,filename,'regularisation parameter')
	return

#=====================================================================
def write_classification_learning_curve_results(alphas,P,R,F1,scores,filename):
	write_classification_sensitivity_results(alphas,P,R,F1,scores,filename,'proportion of total training samples')
	return

#=====================================================================
def write_classification_error_versus_type(max_F1,alphas):
	filename = "classification.all_methods.dat"
	file = open(filename,'w')
	file.write('#\n')
	file.write('# index, method, F1 score, regularisation parameter\n')
	file.write('#\n')
	i=0
	file.write('%d SVM-RBF %18.10e %18.10e\n' % (i, max_F1[i], alphas[i]))
	for i in range(1,len(max_F1)-2):
		file.write('%d SVM-degree-%d %18.10e %18.10e\n' % (i, i, max_F1[i], alphas[i]))
	i+=1
	file.write('%d logistic-L1 %18.10e %18.10e\n' % (i, max_F1[i], alphas[i]))
	i+=1
	file.write('%d logistic-L2 %18.10e %18.10e\n' % (i, max_F1[i], alphas[i]))
	file.close()
	return

#=====================================================================
# EOF
#=====================================================================

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

import m_visualisation as vis
import m_io as io

#=====================================================================
def shuffle_matrices(Z):
	#ZT=Z.T
	#np.random.shuffle(ZT)
	#Z=ZT.T
	np.random.shuffle(Z.T)
	X = Z[:-1,:]
	y = Z[-1,:]
	return X,y

#=====================================================================
def split_classification_data(Xpos, Xneg, ypos, yneg, p=1.0, print_output=False):
	if (print_output):
		print("      Splitting positive classification samples ...")
	num_samples_pos, num_train_samples_pos, num_val_samples_pos, num_test_samples_pos, \
		X_train_pos, X_val_pos, X_test_pos, y_train_pos, y_val_pos, y_test_pos = split_data(Xpos, ypos, p)

	if (print_output):
		print("      Splitting negative classification samples ...")
	num_samples_neg, num_train_samples_neg, num_val_samples_neg, num_test_samples_neg, \
		X_train_neg, X_val_neg, X_test_neg, y_train_neg, y_val_neg, y_test_neg = split_data(Xneg, yneg, p)

	num_samples	    = num_samples_pos	    + num_samples_neg
	num_train_samples   = num_train_samples_pos + num_train_samples_neg
	num_val_samples     = num_val_samples_pos   + num_val_samples_neg
	num_test_samples    = num_test_samples_pos  + num_test_samples_neg
	
	X_train0 = np.hstack((X_train_pos,X_train_neg))
	X_val0   = np.hstack((X_val_pos,  X_val_neg))
	X_test0  = np.hstack((X_test_pos, X_test_neg))

	y_train0 = np.hstack((y_train_pos,y_train_neg))
	y_val0   = np.hstack((y_val_pos,  y_val_neg))
	y_test0  = np.hstack((y_test_pos, y_test_neg))

	X_train, y_train = shuffle_matrices(np.vstack((X_train0,y_train0)))
	X_val,   y_val   = shuffle_matrices(np.vstack((X_val0,  y_val0)))
	X_test,  y_test  = shuffle_matrices(np.vstack((X_test0, y_test0)))

	return num_samples, num_train_samples, num_val_samples, num_test_samples, \
		X_train, X_val, X_test, y_train, y_val, y_test

#=====================================================================
def split_data(X, y, p=1.0, print_output=False):
	num_samples = len(y)					# number of all samples
	N = int(num_samples*0.6)				# maximum number of training samples
	num_train_samples = int(N*p)				# actual number of training samples
	num_val_samples = int(num_samples*0.2)			# number of validation samples
	num_test_samples = num_samples - N - num_val_samples	# number of test samples

	X_train	= X[:,0			:num_train_samples]
	X_val	= X[:,N			:N+num_val_samples]
	X_test	= X[:,N+num_val_samples	:num_samples]

	y_train	= y[  0			:num_train_samples]
	y_val	= y[  N			:N+num_val_samples]
	y_test	= y[  N+num_val_samples	:num_samples]

	if (print_output):
		print("         number of total      samples            = {0}".format(num_samples))
		print("         number of training   samples (max)      = {0}".format(N))
		print("         number of training   samples (selected) = {0}".format(num_train_samples))
		print("         number of validation samples            = {0}".format(num_val_samples))
		print("         number of test       samples            = {0}".format(num_test_samples))

	return num_samples, num_train_samples, num_val_samples, num_test_samples, \
		X_train, X_val, X_test, y_train, y_val, y_test

#=====================================================================
def evaluate_classification_model_performance(y_predict, X, y, print_output=False, file_prefix=''):
	P = 0						# precision  = True positive / (True positive + False positive)
	if (np.sum(y_predict)>0):
		P = np.sum(y_predict*y) / np.sum(y_predict)
	R = 0						# recall     = True positive / (True positive + False negative)
	if (np.sum(y)>0):
		R = np.sum(y_predict*y) / np.sum(y)
	F1 = 0						# F1 score
	if (P+R>0):
		F1 = 2.0*P*R/(P+R)
	if (print_output):
		print("      num positive = {0}".format(np.sum(y)))
		print("      num predicted positive = {0}".format(np.sum(y_predict)))
		print("      num correctly predicted = {0}".format(np.sum(y_predict*y)))
		print("      precision = {0}".format(P))
		print("      recall = {0}".format(R))
		print("      F1 score = {0}".format(F1))
	return P, R, F1

#=====================================================================
def evaluate_regression_model_performance(y_predict, X, y, print_output=False, plot_output=False, file_prefix=''):
	error = np.sum((y_predict-y)**2.0)/len(y)
	if (print_output):
		print("      squared error = {0}".format(error))
	if (plot_output):
		vis.compare_prediction(y, y_predict,file_prefix+'prediction.png')
		vis.compare_error(y, y_predict,file_prefix+'error.png')
	return error

#=====================================================================
def regularise_classification_model(model, Xpos, Xneg, ypos, yneg, alphas, var_name):
	print("\n   Calculating regularisation curves on training and cross validation data sets ...")
	scores_train = np.array(np.zeros(len(alphas), dtype=np.float))
	P_train = np.array(np.zeros(len(alphas), dtype=np.float))
	R_train = np.array(np.zeros(len(alphas), dtype=np.float))
	F1_train = np.array(np.zeros(len(alphas), dtype=np.float))
	scores_val = np.array(np.zeros(len(alphas), dtype=np.float))
	P_val = np.array(np.zeros(len(alphas), dtype=np.float))
	R_val = np.array(np.zeros(len(alphas), dtype=np.float))
	F1_val = np.array(np.zeros(len(alphas), dtype=np.float))

	num_samples, num_train_samples, num_val_samples, num_test_samples, \
		X_train, X_val, X_test, y_train, y_val, y_test = split_classification_data(Xpos, Xneg, ypos, yneg)
	for i,a in enumerate(alphas):
		print("         regularisation parameter = {0}".format(a))
		model.set_params(C=a).fit(X_train.T, y_train)
		scores_train[i] = model.score(X_train.T,y_train)
		scores_val[i] = model.score(X_val.T,y_val)
		P_train[i], R_train[i], F1_train[i] = evaluate_classification_model_performance(model.predict(X_train.T),X_train,y_train)
		P_val[i],   R_val[i],   F1_val[i] = evaluate_classification_model_performance(model.predict(X_val.T),X_val,y_val)
	best_alpha = alphas[np.argmax(F1_val)]
	model.set_params(C=best_alpha).fit(X_train.T, y_train)
	print("      regularisation value  of best fit = {0}".format(best_alpha))

	print("\n   Evaluating performance of model on training data set ...")
	P, R, F1 = evaluate_classification_model_performance(model.predict(X_train.T),X_train,y_train, print_output=True,\
		file_prefix='classification.'+var_name+'.train_')
	print("\n   Evaluating performance of model on validation data set ...")
	P, R, F1 = evaluate_classification_model_performance(model.predict(X_val.T),X_val,y_val,print_output=True,\
		file_prefix='classification.'+var_name+'.validation_')
	print("\n   Evaluating performance of model on test data set ...")
	P, R, F1 = evaluate_classification_model_performance(model.predict(X_test.T),X_test,y_test,print_output=True,\
		file_prefix='classification.'+var_name+'.test_')

	vis.plot_error_versus_alpha(alphas, P_val, P_train, 'precision', 'regularisation parameter', 'logX',\
		'classification.'+var_name+'.regularisation.precision.png')
	vis.plot_error_versus_alpha(alphas, R_val, R_train, 'recall', 'regularisation parameter', 'logX',\
		'classification.'+var_name+'.regularisation.recall.png')
	vis.plot_error_versus_alpha(alphas, F1_val, F1_train, 'F1 score', 'regularisation parameter', 'logX',\
		'classification.'+var_name+'.regularisation.F1_score.png')
	vis.plot_error_versus_alpha(alphas, scores_val, scores_train, 'score', 'regularisation parameter', 'logX',\
		'classification.'+var_name+'.regularisation.score.png')

	io.write_classification_regularisation_results(alphas,P_val,R_val,F1_val,scores_val,\
		'classification.'+var_name+'.regularisation.validation.dat')
	io.write_classification_regularisation_results(alphas,P_train,R_train,F1_train,scores_train,\
		'classification.'+var_name+'.regularisation.train.dat')

	return best_alpha, F1

#=====================================================================
def regularise_regression_model(model, X, y, alphas, var_name):
	print("\n   Calculating regularisation curves on training and cross validation data sets ...")
	scores_train = np.array(np.zeros(len(alphas), dtype=np.float))
	errors_train = np.array(np.zeros(len(alphas), dtype=np.float))
	scores_val = np.array(np.zeros(len(alphas), dtype=np.float))
	errors_val = np.array(np.zeros(len(alphas), dtype=np.float))

	num_samples, num_train_samples, num_val_samples, num_test_samples, \
		X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y)
	for i,a in enumerate(alphas):
		print(np.shape(X_train.T),np.shape(y_train))
		model.set_params(alpha=a).fit(X_train.T, y_train)
		scores_train[i] = model.score(X_train.T,y_train)
		scores_val[i] = model.score(X_val.T,y_val)
		errors_train[i] = evaluate_regression_model_performance(model.predict(X_train.T),X_train,y_train)
		errors_val[i] = evaluate_regression_model_performance(model.predict(X_val.T),X_val,y_val)

	best_alpha = alphas[np.argmin(errors_val)]
	model.set_params(alpha=best_alpha).fit(X_train.T, y_train)
	print("      regularisation value of best fit = {0}".format(best_alpha))
	print("      model coefficients = {0}\n".format(model.coef_))

	print("\n   Evaluating performance of model on training data set ...")
	error = evaluate_regression_model_performance(model.predict(X_train.T),X_train,y_train, print_output=True,plot_output=True,\
		file_prefix='regression.'+var_name+'.train_')
	print("\n   Evaluating performance of model on validation data set ...")
	error = evaluate_regression_model_performance(model.predict(X_val.T),X_val,y_val,print_output=True,plot_output=True,\
		file_prefix='regression.'+var_name+'.validation_')
	print("\n   Evaluating performance of model on test data set ...")
	error = evaluate_regression_model_performance(model.predict(X_test.T),X_test,y_test,print_output=True,plot_output=True,\
		file_prefix='regression.'+var_name+'.test_')

	vis.plot_error_versus_alpha(alphas, errors_val, errors_train, 'squared error', 'regularisation parameter', 'logXY',\
		'regression.'+var_name+'.regularisation.sq_error.png')
	vis.plot_error_versus_alpha(alphas, scores_val, scores_train, 'score', 'regularisation parameter', 'logX',\
		'regression.'+var_name+'.regularisation.score.png')

	io.write_regression_regularisation_results(alphas,errors_val,scores_val,\
		'regression.'+var_name+'.regularisation.validation.dat')
	io.write_regression_regularisation_results(alphas,errors_train,scores_train,\
		'regression.'+var_name+'.regularisation.train.dat')

	return best_alpha, error

#=====================================================================
def calculate_classification_learning_curves(model, Xpos, Xneg, ypos, yneg, var_name):
	print("\n   Calculating learning curves on cross validation data set ...")
	proportion_of_samples = np.linspace(0.2, 1.0, 32)
	scores_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	P_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	R_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	F1_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	scores_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	P_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	R_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	F1_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))

	for i,p in enumerate(proportion_of_samples):
		num_samples, num_train_samples, num_val_samples, num_test_samples, \
			X_train, X_val, X_test, y_train, y_val, y_test = split_classification_data(Xpos, Xneg, ypos, yneg, p)
		model.fit(X_train.T, y_train)
		scores_train_lc[i] = model.score(X_train.T,y_train)
		scores_val_lc[i] = model.score(X_val.T,y_val)
		P_train_lc[i],R_train_lc[i],F1_train_lc[i]=evaluate_classification_model_performance(model.predict(X_train.T),X_train,y_train)
		P_val_lc[i],R_val_lc[i],F1_val_lc[i]=evaluate_classification_model_performance(model.predict(X_val.T),X_val,y_val)
		del  X_train, X_val, X_test, y_train, y_val, y_test

	vis.plot_error_versus_alpha(proportion_of_samples, P_val_lc, P_train_lc, 'precision', \
		'proportion of total training samples', 'linear',\
		'classification.'+var_name+'.learning_curves.precision.png')
	vis.plot_error_versus_alpha(proportion_of_samples, R_val_lc, R_train_lc, 'recall', \
		'proportion of total training samples', 'linear',\
		'classification.'+var_name+'.learning_curves.recall.png')
	vis.plot_error_versus_alpha(proportion_of_samples, F1_val_lc, F1_train_lc, 'F1 score', \
		'proportion of total training samples', 'linear',\
		'classification.'+var_name+'.learning_curves.F1_score.png')
	vis.plot_error_versus_alpha(proportion_of_samples, scores_val_lc, scores_train_lc, 'score', \
		'proportion of total training samples', 'linear',\
		'classification.'+var_name+'.learning_curves.score.png')

	print("\n   Need to later undertake manual error analysis on the cross validation set")
	print("   to find commonalities between poorly predicted samples.")
	return

#=====================================================================
def calculate_regression_learning_curves(model, X, y, var_name):
	print("\n   Calculating learning curves on cross validation data set ...")
	proportion_of_samples = np.linspace(0.2, 1.0, 32)
	scores_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	errors_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	scores_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	errors_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))

	for i,p in enumerate(proportion_of_samples):
		num_samples, num_train_samples, num_val_samples, num_test_samples, \
			X_train, X_val, X_test, y_train, y_val, y_test = split_data(X,y,p)
		model.fit(X_train.T, y_train)
		scores_val_lc[i] = model.score(X_val.T,y_val) 
		scores_train_lc[i] = model.score(X_train.T,y_train)
		errors_train_lc[i] = evaluate_regression_model_performance(model.predict(X_train.T),X_train,y_train)
		errors_val_lc[i] = evaluate_regression_model_performance(model.predict(X_val.T),X_val,y_val)
		del  X_train, X_val, X_test, y_train, y_val, y_test

	vis.plot_error_versus_alpha(proportion_of_samples, errors_val_lc, errors_train_lc, 'squared error', \
		'proportion of total training samples','linear', 'regression.'+var_name+'.learning_curves.sq_error.png')
	vis.plot_error_versus_alpha(proportion_of_samples, scores_val_lc, scores_train_lc, 'score', \
		'proportion of total training samples','linear', 'regression.'+var_name+'.learning_curves.score.png')

	print("\n   Need to later undertake manual error analysis on the cross validation set")
	print("   to find commonalities between poorly predicted samples.")
	return

#=====================================================================
def perform_lasso_regression(X, y, var_name):
	print("\n\nPerforming Lasso Regression on variable {0}".format(var_name))
	model = skl.linear_model.Lasso()	#model = skl.linear_model.Ridge()
	model.set_params(max_iter=100000,tol=1e-3)

	filename_prefix = var_name+'.lasso'
	alphas = np.logspace(-5, 0, 32)
	best_alpha, min_error = regularise_regression_model(model, X, y, alphas, filename_prefix)

	model.set_params(alpha=best_alpha)
	calculate_regression_learning_curves(model, X, y, filename_prefix)
	return best_alpha, min_error

#=====================================================================
def perform_logistic_classification(Xpos, Xneg, ypos, yneg, var_name, L):
	print("\n\nPerforming Logistic Classification on variable {0} using {1} regularisation".format(var_name,L))
	model = skl.linear_model.LogisticRegression(penalty=L,tol=1e-6)
	filename_prefix = var_name+'.'+L+'.logistic'

	alphas = np.logspace(-1, 1, 33)
	best_alpha, max_F1 = regularise_classification_model(model, Xpos, Xneg, ypos, yneg, alphas, filename_prefix)
	print("      model coefficients = {0}\n".format(model.coef_))

	model.set_params(C=best_alpha)
	calculate_classification_learning_curves(model, Xpos, Xneg, ypos, yneg, filename_prefix)
	return best_alpha, max_F1

#=====================================================================
def perform_svm_classification(Xpos, Xneg, ypos, yneg, var_name, D):
	print("\n\nPerforming Support Vector Machine Classification of degree {1} on variable {0}".format(var_name,D))
	if (D==0):
		model = SVC(kernel='rbf',tol=1e-4)
		filename_prefix = var_name+'.svm_rbf'
	else:
		model = SVC(kernel='poly',tol=1e-4,degree=D)
		filename_prefix = var_name+'.svm_degree'+'%02d'%(D)

	alphas = np.logspace(-1, 1, 33)
	best_alpha, max_F1 = regularise_classification_model(model, Xpos, Xneg, ypos, yneg, alphas, filename_prefix)

	model.set_params(C=best_alpha)
	calculate_classification_learning_curves(model, Xpos, Xneg, ypos, yneg, filename_prefix)
	return best_alpha, max_F1

#=====================================================================
# EOF
#=====================================================================

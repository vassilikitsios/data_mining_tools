#=====================================================================
# import libraries
import numpy as np

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
# EOF
#=====================================================================

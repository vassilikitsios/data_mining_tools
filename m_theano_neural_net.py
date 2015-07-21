#=====================================================================
"""A multilayer perceptron is a logistic regressor where instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear activation function (usually tanh or sigmoid).
Theano code modified from theano tutorials."""
__docformat__ = 'restructedtext en'

#=====================================================================
# import libraries
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T

import m_machine_learning as ml
from m_theano_logistic_regression import LogisticRegression
import m_visualisation as vis
import m_io as io

#=====================================================================
# to do:
	# predict output from test set and evaluate error measures

#=====================================================================
class HiddenLayer(object):

	#---------------------------------------------------------------------
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
		""" Typical hidden layer of a MLP: units are fully-connected and have sigmoidal activation function.
		Weight matrix W is of shape (n_in,n_out) and the bias vector b is of shape (n_out,).
		By default the nonlinearity used here is tanh, with the hidden unit activation given by: tanh(dot(input,W) + b).
			rng (numpy.random.RandomState) : a random number generator used to initialize weights
			input (theano.tensor.dmatrix) : a symbolic tensor of shape (n_examples, n_in)
			n_in (int) : dimensionality of input
			n_out (int) : number of hidden units
			activation (theano.Op or function) : Non linearity to be applied in the hidden layer """
		self.input = input

		""" `W` is initialized with `W_values` which is uniformely sampled from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
	        for tanh activation function the output of uniform if converted using asarray to dtype
		theano.config.floatX so that the code is runable on GPU
	        Optimal initialization of weights is dependent on the activation function used (among other things).
        	One should use 4 times larger initial weights for sigmoid compared to tanh. """
		if W is None:
			W_values = np.asarray( rng.uniform( low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), \
				size=(n_in, n_out)), dtype=theano.config.floatX)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b
		lin_output = T.dot(input, self.W) + self.b
		self.output = ( lin_output if activation is None else activation(lin_output))

		# parameters of the model
		self.params = [self.W, self.b]
		return

#=====================================================================
class MLP(object):
	"""Multi-Layer Perceptron (MLP) Class
		A MLP is a feedforward artificial neural network model that has one layer or more of hidden units and nonlinear activations.
		Intermediate layers usually have as activation function tanh or the sigmoid function (defined here by a ``HiddenLayer`` class)
		The top layer is a softmax layer (defined here by a ``LogisticRegression`` class).  """

	#---------------------------------------------------------------------
	def __init__(self, rng, input, n_in, n_hidden, n_out):
		"""Initialize the parameters for the multilayer perceptron
		        rng (numpy.random.RandomState) : a random number generator used to initialize weights
			input (theano.tensor.TensorType) : symbolic variable that describes the input of the architecture (one minibatch)
			n_in (int) : number of input units, the dimension of the space in which the datapoints lie
        		n_hidden (int) : number of hidden units
			n_out (int) : number of output units, the dimension of the space in which the labels lie
       		By default we have a MLP with one hidden layer with a tanh activation function connected to 
		the LogisticRegression layer the activation function can be replaced by sigmoid or any other nonlinear function.  """

		self.hiddenLayer = HiddenLayer( rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)

		# The logistic regression layer gets as input the hidden units of the hidden layer
		self.logRegressionLayer = LogisticRegression( input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)

		# Regularisation functions for L1 and L2 norm
		self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
		self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

		# negative log likelihood of the MLP is given by the negative log likelihood of the output of the model,
		# computed in the logistic regression layer (also for the error measures)
		self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
		self.errors = self.logRegressionLayer.errors
		#self.evaluate_model_performance = self.logRegressionLayer.evaluate_model_performance

		# the parameters of the model are the parameters of the two layer it is made out of
		self.params = self.hiddenLayer.params + self.logRegressionLayer.params
		return
	#---------------------------------------------------------------------

#=====================================================================
def build_model(data_train_set_x, data_valid_set_x, data_test_set_x, data_train_set_y, data_valid_set_y, data_test_set_y, \
	learning_rate=0.1, L1_reg=0.001, L2_reg=0.001, n_epochs=2000, batch_size=100, n_hidden=24):
	""" Stochastic gradient descent optimization for a multilayer perceptron
		learning_rate (float) : learning rate factor used for the stochastic gradient
		L1_reg (float) : L1-norm's weight when added to the cost 
		L2_reg (float) : L2-norm's weight when added to the cost
		n_epochs (int) : maximal number of epochs to run the optimizer """

	#---------------------------------------------------------------------
	print('Assigning data to theano tensor data types ...')
	train_set_x = theano.shared(np.asarray(data_train_set_x.T, dtype=theano.config.floatX), borrow=True)
	valid_set_x = theano.shared(np.asarray(data_valid_set_x.T, dtype=theano.config.floatX), borrow=True)
	test_set_x  = theano.shared(np.asarray(data_test_set_x.T,  dtype=theano.config.floatX), borrow=True)

	""" Original code only works for outputs of dimenions > 1, with each dimension turing on the associated class.
	The original output vector has only one dimension consisting entires of 1 or 0.
	An additional vector of 1 minus the original output is stacked to the original vector using hstack. """
	train_set_y = T.cast( theano.shared(np.asarray(np.hstack((data_train_set_y,1-data_train_set_y)), \
		dtype=theano.config.floatX), borrow=True), 'int32')
	valid_set_y = T.cast( theano.shared(np.asarray(np.hstack((data_valid_set_y,1-data_valid_set_y)), \
		dtype=theano.config.floatX), borrow=True), 'int32')
	test_set_y  = T.cast( theano.shared(np.asarray(np.hstack((data_test_set_y, 1-data_test_set_y )), \
		dtype=theano.config.floatX), borrow=True), 'int32')

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	#---------------------------------------------------------------------
	print('Building the model ...')

	# allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

	rng = np.random.RandomState(1234)

	# construct the MLP class
	classifier = MLP( rng=rng, input=x, n_in=12, n_hidden=n_hidden, n_out=2)

	# the cost we minimize during training is the negative log likelihood of the model
	# plus the regularization terms (L1 and L2); cost is expressed here symbolically
	cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

	# compiling a Theano function that computes the mistakes that are made by the model on a minibatch
	test_model = theano.function( inputs=[index], outputs=classifier.errors(y), givens={\
		x: test_set_x[index * batch_size:(index + 1) * batch_size],\
		y: test_set_y[index * batch_size:(index + 1) * batch_size] })

	validate_model = theano.function( inputs=[index], outputs=classifier.errors(y), givens={\
		x: valid_set_x[index * batch_size:(index + 1) * batch_size],
		y: valid_set_y[index * batch_size:(index + 1) * batch_size] })

	# compute the gradient of cost with respect to theta (sotred in params) the resulting gradients will be stored in a list gparams
	gparams = [T.grad(cost, param) for param in classifier.params]

	# specify how to update the parameters of the model as a list of (variable, update expression) pairs
	updates = [ (param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams) ]

	# compiling a Theano function `train_model` that returns the cost,
	# but in the same time updates the parameter of the model based on the rules defined in `updates`
	train_model = theano.function( inputs=[index], outputs=cost, updates=updates, givens={\
		x: train_set_x[index * batch_size: (index + 1) * batch_size],\
		y: train_set_y[index * batch_size: (index + 1) * batch_size] })

	train_model_errors = theano.function( inputs=[index], outputs=classifier.errors(y), givens={\
		x: train_set_x[index * batch_size: (index + 1) * batch_size],\
		y: train_set_y[index * batch_size: (index + 1) * batch_size] })

        #---------------------------------------------------------------------
	print('Training the model ...')
	patience = 10000						# look as this many examples regardless
	patience_increase = 2						# wait this much longer when a new best is found
	improvement_threshold = 0.995					# a relative improvement of this much is considered significant
	validation_frequency = min(n_train_batches, patience / 2)	# go through this many minibatche before checking the network
									# on the validation set; in this case we check every epoch
	best_validation_loss = np.inf
	best_iter = 0
	test_score = 0.0
	start_time = time.clock()

	train_performance_measures          = np.array(np.zeros((n_train_batches,4), dtype=np.float))
	validation_performance_measures     = np.array(np.zeros((n_valid_batches,4), dtype=np.float))
	test_performance_measures           = np.array(np.zeros((n_test_batches,4), dtype=np.float))
	avg_train_performance_measures      = np.array(np.zeros((4), dtype=np.float))
	avg_validation_performance_measures = np.array(np.zeros((4), dtype=np.float))
	avg_test_performance_measures       = np.array(np.zeros((4), dtype=np.float))

	epoch = 0
	done_looping = False
	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1

		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
			iter = (epoch - 1) * n_train_batches + minibatch_index

			if (iter + 1) % validation_frequency == 0:
				for i in xrange(n_valid_batches):
					validation_performance_measures[i,:] = validate_model(i)[:]
				this_validation_loss = np.mean(validation_performance_measures[:,0])
				print( 'epoch %i, minibatch %i/%i, validation error %f %%' \
					% ( epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

				if this_validation_loss < best_validation_loss:
					if ( this_validation_loss < best_validation_loss * improvement_threshold):
						patience = max(patience, iter * patience_increase)
					best_validation_loss = this_validation_loss
					best_iter = iter
					for i in xrange(n_test_batches):
						test_performance_measures[i,:] = test_model(i)[:]
					test_score = np.mean(test_performance_measures[:,0])
					print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') \
						% (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
			if patience <= iter:
				done_looping = True
				break

	end_time = time.clock()
	print(('Optimization complete. Best validation score of %f %% obtained at iteration %i, with test performance %f %%') \
		% (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)) 

	for i in xrange(n_train_batches):
		train_performance_measures[i,:] = train_model_errors(i)
	for i in xrange(n_valid_batches):
		validation_performance_measures[i,:] = validate_model(i)
	for i in xrange(n_test_batches):
		test_performance_measures[i,:] = test_model(i)

	for i in xrange(4):
		avg_train_performance_measures[i]      = np.mean(train_performance_measures[:,i])
		avg_validation_performance_measures[i] = np.mean(validation_performance_measures[:,i])
		avg_test_performance_measures[i]       = np.mean(test_performance_measures[:,i])

	return avg_train_performance_measures, avg_validation_performance_measures, avg_test_performance_measures

#=====================================================================
def regularise_classification_model(L1_flag, L2_flag, Xpos, Xneg, ypos, yneg, alphas, var_name):
	print("\n   Calculating regularisation curves on training and cross validation data sets ...")
	error_train = np.array(np.zeros(len(alphas), dtype=np.float))
	P_train = np.array(np.zeros(len(alphas), dtype=np.float))
	R_train = np.array(np.zeros(len(alphas), dtype=np.float))
	F1_train = np.array(np.zeros(len(alphas), dtype=np.float))
	error_val = np.array(np.zeros(len(alphas), dtype=np.float))
	P_val = np.array(np.zeros(len(alphas), dtype=np.float))
	R_val = np.array(np.zeros(len(alphas), dtype=np.float))
	F1_val = np.array(np.zeros(len(alphas), dtype=np.float))

	num_samples, num_train_samples, num_val_samples, num_test_samples, \
		X_train, X_val, X_test, y_train, y_val, y_test = ml.split_classification_data(Xpos, Xneg, ypos, yneg)

	for i,a in enumerate(alphas):
		print("\n         regularisation parameter = {0}\n".format(a))
		avg_train_performance_measures, avg_validation_performance_measures, avg_test_performance_measures\
			= build_model( X_train, X_val, X_test, y_train, y_val, y_test, L1_reg=L1_flag*a, L2_reg=L2_flag*a)
		error_train[i]  = avg_train_performance_measures[0]
		P_train[i]      = avg_train_performance_measures[1]
		R_train[i]      = avg_train_performance_measures[2]
		F1_train[i]     = avg_train_performance_measures[3]
		error_val[i]    = avg_validation_performance_measures[0]
		P_val[i]        = avg_validation_performance_measures[1]
		R_val[i]        = avg_validation_performance_measures[2]
		F1_val[i]       = avg_validation_performance_measures[3]

	best_alpha = alphas[np.argmax(F1_val)]
	#best_alpha = alphas[np.argmin(error_val)]
	print("      regularisation value  of best fit = {0}".format(best_alpha))
	avg_train_performance_measures, avg_validation_performance_measures, avg_test_performance_measures\
		= build_model( X_train, X_val, X_test, y_train, y_val, y_test, L1_reg=L1_flag*best_alpha, L2_reg=L2_flag*best_alpha)
	print("train performance measures: {0}".format(avg_train_performance_measures))
	print("validation performance measures: {0}".format(avg_validation_performance_measures))
	print("test performance measures: {0}".format(avg_test_performance_measures))

	vis.plot_error_versus_alpha(alphas, error_val, error_train, 'error', 'regularisation parameter', 'logX',\
		'classification.'+var_name+'.regularisation.error.png')
	vis.plot_error_versus_alpha(alphas, P_val, P_train, 'precision', 'regularisation parameter', 'logX',\
		'classification.'+var_name+'.regularisation.precision.png')
	vis.plot_error_versus_alpha(alphas, R_val, R_train, 'recall', 'regularisation parameter', 'logX',\
		'classification.'+var_name+'.regularisation.recall.png')
	vis.plot_error_versus_alpha(alphas, F1_val, F1_train, 'F1 score', 'regularisation parameter', 'logX',\
		'classification.'+var_name+'.regularisation.F1_score.png')

	io.write_classification_regularisation_results(alphas,P_val,R_val,F1_val,error_val,\
		'classification.'+var_name+'.regularisation.validation.dat')
	io.write_classification_regularisation_results(alphas,P_train,R_train,F1_train,error_train,\
		'classification.'+var_name+'.regularisation.train.dat')

	return best_alpha

#=====================================================================
def calculate_classification_learning_curves(L1_flag, L2_flag, best_alpha, Xpos, Xneg, ypos, yneg, var_name):
	print("\n   Calculating learning curves on cross validation data set ...")
	proportion_of_samples = np.linspace(0.2, 1.0, 32)
	error_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	P_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	R_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	F1_train_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	error_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	P_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	R_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))
	F1_val_lc = np.array(np.zeros(len(proportion_of_samples), dtype=np.float))

	for i,p in enumerate(proportion_of_samples):
		print("\n         proportional of total samples = {0}\n".format(p))
		num_samples, num_train_samples, num_val_samples, num_test_samples, \
			X_train, X_val, X_test, y_train, y_val, y_test = ml.split_classification_data(Xpos, Xneg, ypos, yneg, p)
		avg_train_performance_measures, avg_validation_performance_measures, avg_test_performance_measures\
			= build_model( X_train, X_val, X_test, y_train, y_val, y_test, L1_reg=L1_flag*best_alpha, L2_reg=L2_flag*best_alpha)
		error_train_lc[i]  = avg_train_performance_measures[0]
		P_train_lc[i]      = avg_train_performance_measures[1]
		R_train_lc[i]      = avg_train_performance_measures[2]
		F1_train_lc[i]     = avg_train_performance_measures[3]
		error_val_lc[i]    = avg_validation_performance_measures[0]
		P_val_lc[i]        = avg_validation_performance_measures[1]
		R_val_lc[i]        = avg_validation_performance_measures[2]
		F1_val_lc[i]       = avg_validation_performance_measures[3]
		del  X_train, X_val, X_test, y_train, y_val, y_test

	vis.plot_error_versus_alpha(proportion_of_samples, error_val_lc, error_train_lc, 'error', \
		'proportion of total training samples', 'linear',\
		'classification.'+var_name+'.learning_curves.error.png')
	vis.plot_error_versus_alpha(proportion_of_samples, P_val_lc, P_train_lc, 'precision', \
		'proportion of total training samples', 'linear',\
		'classification.'+var_name+'.learning_curves.precision.png')
	vis.plot_error_versus_alpha(proportion_of_samples, R_val_lc, R_train_lc, 'recall', \
		'proportion of total training samples', 'linear',\
		'classification.'+var_name+'.learning_curves.recall.png')
	vis.plot_error_versus_alpha(proportion_of_samples, F1_val_lc, F1_train_lc, 'F1 score', \
		'proportion of total training samples', 'linear',\
		'classification.'+var_name+'.learning_curves.F1_score.png')

	print("\n   Need to later undertake manual error analysis on the cross validation set")
	print("   to find commonalities between poorly predicted samples.")
	return

#=====================================================================
def perform_classification(Xpos, Xneg, ypos, yneg, var_name, L):
	print("\n\nPerforming Neural Network Classification on variable {0} using {1} regularisation".format(var_name,L))
	filename_prefix = var_name+'.'+L+'.neural_net'
	if (L=="l1"):
		L1_flag=1
		L2_flag=0
	if (L=="l2"):
		L1_flag=0
		L2_flag=1
	alphas = np.logspace(-3, -1, 21)
	best_alpha = regularise_classification_model(L1_flag, L2_flag, Xpos, Xneg, ypos, yneg, alphas, filename_prefix)
	calculate_classification_learning_curves(L1_flag, L2_flag, best_alpha, Xpos, Xneg, ypos, yneg, filename_prefix)
	return

#=====================================================================
# EOF
#=====================================================================

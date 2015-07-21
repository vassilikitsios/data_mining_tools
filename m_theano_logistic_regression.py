#=====================================================================
""" Logistic regression is a probabilistic, linear classifier.

Theano code modified from theano tutorials.

It is parametrized by a weight matrix :math:`W` and a bias vector :math:`b`.
Classification is done by projecting data points onto a set of hyperplanes,
the distance to which is used to determine a class membership probability.
The output of the model or prediction is then done by taking the argmax of the vector whose i'th element is P(Y=i|x)."""
__docformat__ = 'restructedtext en'

#=====================================================================
# import libraries
import os
import sys
import time
import numpy as np

import theano
import theano.tensor as T
from theano import function

import m_machine_learning as ml

#=====================================================================
# to do:
	# add in regularisation to the cost function

#=====================================================================
class LogisticRegression(object):
	"""Multi-class Logistic Regression Class
		The logistic regression is fully described by a weight matrix :math:`W` and bias vector :math:`b`.
		Classification is done by projecting data points onto a set of hyperplanes,
		the distance to which is used to determine a class membership probability.
	"""

	#---------------------------------------------------------------------
	def __init__(self, input, n_in, n_out):
		""" Initialize the parameters of the logistic regression
			input (theano.tensor.TensorType) : symbolic variable that describes the input of the architecture (one minibatch)
			n_in (int) : number of input units, the dimension of the space in which the datapoints lie
			n_out (int) : number of output units, the dimension of the space in which the labels lie """
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		self.W = theano.shared( value=np.zeros( (n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)

		# initialize the baises b as a vector of n_out 0s
		self.b = theano.shared( value=np.zeros( (n_out,), dtype=theano.config.floatX), name='b', borrow=True)

		# symbolic expression for computing the matrix of class-membership probabilities. Where:
		# W is a matrix where column-k represent the separation hyper plain for class-k
		# x is a matrix where row-j  represents input training sample-j
		# b is a vector where element-k represent the free parameter of hyper plain-k
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		# symbolic description of how to compute prediction as class whose probability is maximal
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		# parameters of the model
		self.params = [self.W, self.b]
		return

	#---------------------------------------------------------------------
	def negative_log_likelihood(self, y):
		"""Return the mean of the negative log-likelihood of the prediction of this model under a given target distribution.
			.. math::
			\frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
			\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
			\ell (\theta=\{W,b\}, \mathcal{D})

			y (theano.tensor.TensorType) : corresponds to a vector that gives for each example the correct label

	        Note: we use the mean instead of the sum so that the learning rate is less dependent on the batch size

		y.shape[0] is (symbolically) the number of rows in y, i.e., number of examples (call it n) in the minibatch
		T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
		Log-Probabilities (call it LP) with one row per example and one column per class LP[T.arange(y.shape[0]),y] is a vector
		v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
		the mean (across minibatch examples) of the elements in v, i.e., the mean log-likelihood across the minibatch. """
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	#---------------------------------------------------------------------
	def errors(self, y, print_output=False):
		# check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))

		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			num_positive = T.cast(T.sum(T.eq(y,1)),'float64')
			num_predicted_positive = T.cast(T.sum(T.eq(self.y_pred,1)),'float64')
			num_correctly_predicted = T.cast(T.sum(T.eq(self.y_pred*y,1)),'float64')

			P = T.cast(0.0,'float64')	# precision  = True positive / (True positive + False positive)
			if (T.gt(num_predicted_positive,0.0)):
				P = T.cast(num_correctly_predicted / num_predicted_positive,'float64')

			R = T.cast(0.0,'float64')	# recall     = True positive / (True positive + False negative)
			if (T.gt(num_positive,0.0)):
				R = T.cast(num_correctly_predicted / num_positive,'float64')

			F1 = T.cast(0.0,'float64')	# F1 score
			if (T.gt(P+R,0.0)):
				F1 = 2.0*P*R/(P+R)

			if (print_output):
				print("      num positive = {0}".format( num_positive ) )
				print("      num predicted positive = {0}".format( num_predicted_positive ) )
				print("      num correctly predicted = {0}".format( num_correctly_predicted ) )
				print("      precision = {0}".format(P))
				print("      recall = {0}".format(R))
				print("      F1 score = {0}".format(F1))
			return [T.mean(T.neq(self.y_pred, y)), P, R, F1]

		else:
			raise NotImplementedError()
		return

	#---------------------------------------------------------------------

#=====================================================================
def perform_classification(Xpos, Xneg, ypos, yneg, learning_rate=0.13, n_epochs=1000, batch_size=100):
	""" Stochastic gradient descent optimization for logistic regression
		learning_rate (float): learning rate factor used for the stochastic gradient
		n_epochs (int): maximal number of epochs to run the optimizer """

	#---------------------------------------------------------------------
	print('Assigning data to theano tensor data types ...')
	num_samples, num_train_samples, num_val_samples, num_test_samples, \
		data_train_set_x, data_valid_set_x, data_test_set_x, data_train_set_y, data_valid_set_y, data_test_set_y \
		= ml.split_classification_data(Xpos, Xneg, ypos, yneg)

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
	index = T.lscalar()  # index to a minibatch

	# generate symbolic variables for input (x and y represent a minibatch)
	x = T.matrix('x')  # data, presented as rasterized images
	y = T.ivector('y')  # labels, presented as 1D vector of [int] labels 

	# construct the logistic regression class
	classifier = LogisticRegression(input=x, n_in=12, n_out=2)

	# the cost we minimize during training is the negative log likelihood of the model in symbolic format
	cost = classifier.negative_log_likelihood(y)

	# compiling a Theano function that computes the mistakes that are made by the model on a minibatch
	test_model = theano.function( inputs=[index], outputs=classifier.errors(y), givens={\
		x: test_set_x[index * batch_size: (index + 1) * batch_size],\
		y: test_set_y[index * batch_size: (index + 1) * batch_size] })

	validate_model = theano.function( inputs=[index], outputs=classifier.errors(y), givens={\
		x: valid_set_x[index * batch_size: (index + 1) * batch_size],\
		y: valid_set_y[index * batch_size: (index + 1) * batch_size] })

	# compute the gradient of cost with respect to theta = (W,b)
	g_W = T.grad(cost=cost, wrt=classifier.W)
	g_b = T.grad(cost=cost, wrt=classifier.b)

	# specify how to update the parameters of the model as a list of (variable, update expression) pairs.
	updates = [(classifier.W, classifier.W - learning_rate * g_W), (classifier.b, classifier.b - learning_rate * g_b)]

	# compiling a Theano function `train_model` that returns the cost,
	# but in the same time updates the parameter of the model based on the rules defined in `updates`
	train_model = theano.function( inputs=[index], outputs=cost, updates=updates, givens={\
		x: train_set_x[index * batch_size: (index + 1) * batch_size],\
		y: train_set_y[index * batch_size: (index + 1) * batch_size] })

	#---------------------------------------------------------------------
	print('Training the model ...')
	patience = 5000			# look as this many examples regardless
	patience_increase = 2		# wait this much longer when a new best is found
	improvement_threshold = 0.995 	# a relative improvement of this much is considered significant
	validation_frequency = min(n_train_batches, patience / 2) 	# go through this many minibatches before checking the network 
									# on the validation set; in this case we check every epoch
	best_validation_loss = np.inf
	test_score = 0.0
	start_time = time.clock()

	done_looping = False
	epoch = 0
	while (epoch < n_epochs) and (not done_looping):
		epoch += 1
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
			iter = (epoch - 1) * n_train_batches + minibatch_index	# iteration number
			if (iter + 1) % validation_frequency == 0:
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
				this_validation_loss = np.mean(validation_losses)
				print( ('epoch %i, minibatch %i/%i, validation error %f %%') \
					% ( epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))
				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					#improve patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss * improvement_threshold:
						patience = max(patience, iter * patience_increase)
					best_validation_loss = this_validation_loss
					# test it on the test set
					test_losses = [test_model(i) for i in xrange(n_test_batches)]
					test_score = np.mean(test_losses)
					print( ('     epoch %i, minibatch %i/%i, test error of' ' best model %f %%') \
						% ( epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
			if patience <= iter:
				done_looping = True
				break
	end_time = time.clock()

	print( ( 'Optimization complete with best validation score of %f %%,' 'with test performance %f %%') \
		% (best_validation_loss * 100., test_score * 100.))
	print('The code run for {0} epochs, with {1} epochs/sec'.format(epoch, 1. * epoch / (end_time - start_time)))
	print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))

	return

#=====================================================================
# EOF
#=====================================================================

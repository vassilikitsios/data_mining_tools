#=====================================================================
# regression.py
'''
Contains class RegressionAlgorythm which inherents its properties 
from base class SupervisedMachineLearningAlgorythm.
'''

import numpy as np
import sys

import sklearn as skl
from sklearn.linear_model import LinearRegression

from libMachineLearning.machine_learning import SupervisedMachineLearningAlgorythm
import libVisualisation.visualisation as vis


#=====================================================================
class RegressionAlgorythm(SupervisedMachineLearningAlgorythm):
    '''
    Class to build and test regression machine learning algorythms.
    '''
    
#---------------------------------------------------------------------        
    def __init__(self,X,y,var_name,regression_type,file_format,\
        number_of_regularisation_levels, min_log_of_regularisation_level, max_log_of_regularisation_level,
        number_of_proportions_of_samples, min_proportion_of_samples, max_proportion_of_samples=1.0,
        max_iterations=100000,tolerance=1e-3):
        '''
        Initiates input data X and output data y.
        Regression model is assigned and master class SupervisedMachineLearningAlgorythm
        initiated.
        '''
        self.X = X
        self.y = y
        self.X, self.y = self.shuffle_matrices(np.vstack((self.X, self.y)))

        if (regression_type=="lasso"):
            print("\n\nPerforming Lasso Regression on variable {0}".format(var_name))
            self.model = skl.linear_model.Lasso()
            self.model.set_params(max_iter=max_iterations,tol=tolerance)
            self.filename_prefix = 'regression.'+var_name+'.lasso'
        elif (regression_type=="ridge"):
            print("\n\nPerforming Ridge Regression on variable {0}".format(var_name))
            self.model = skl.linear_model.Ridge()
            self.model.set_params(max_iter=max_iterations,tol=tolerance)
            self.filename_prefix = 'regression.'+var_name+'.ridge'
        else:
            print("\n\nRegression type not valid: lasso; ridge")
            sys.exit()  
        
        SupervisedMachineLearningAlgorythm.__init__(self, var_name, file_format,\
            number_of_regularisation_levels, min_log_of_regularisation_level, max_log_of_regularisation_level,
            number_of_proportions_of_samples, min_proportion_of_samples, max_proportion_of_samples) 
 
        return

#---------------------------------------------------------------------
    def split_data(self, p0=1.0, print_output=False):
        '''
        Splits data into training, validation and testing sets.
        '''
        return self.split_this_data(self.X, self.y, p0, print_output)

#---------------------------------------------------------------------    
    def build_model(self, X0, y0, regularisation_level):
        '''
        Assigns regularisation level and builds model linking input data (X0)
        to output data (y0).
        '''
        self.model.set_params(alpha=regularisation_level).fit(X0.T, y0)

#---------------------------------------------------------------------            
    def evaluate_model_performance(self, X0, y0, parameter, print_output=False, plot_output=False, \
        file_prefix=''):
        '''
        Evaluates the score and squared error of the regression prediction,
        and returns as a tuple including the input parameter.
        For regularisation studies, the input parameter is the regularisation level.
        For learning curves, the input parameter is the proportion of total sampels.
        '''
        score = self.model.score(X0.T,y0)
        y_predict = self.model.predict(X0.T)
        error = np.sum((y_predict-y0)**2.0)/len(y0)
        if (print_output):
            print("      squared error = {0}".format(error))
        if (plot_output):
            vis.compare_prediction(y0, y_predict,file_prefix+'prediction.png')
            vis.compare_error(y0, y_predict,file_prefix+'error.png')
        return (parameter, score, error)

#---------------------------------------------------------------------            
    def calculate_best_regularisation_level(self):
        '''
        The best regularisation level for classification studies is defined
        as that with the minimum squared error.
        '''
        min_error = 1.0e10
        best_regularisation_level = 0.0
        for regularisation_level, score, error in self.validation_performance_regularisation:
            if (error<min_error):
                min_error = error
                best_regularisation_level = regularisation_level
        return best_regularisation_level

#---------------------------------------------------------------------    
    def plot_output(self,validation_performance,training_performance,x_label,plot_type,file_suffix):
        '''
        Plot each of the performance measures for both the training and validation
        data sets against the associated input parameter (eg: regularisation level).
        '''
        parameters = []
        score_train = []
        error_train = []
        for parameter, score, error in training_performance:
            parameters.append(parameter)
            score_train.append(score)
            error_train.append(error)

        score_val = []
        error_val = []
        for parameter, score, error in validation_performance:
            score_val.append(score)
            error_val.append(error)

        vis.plot_performance(parameters, error_val, error_train, 'squared error', \
            x_label, plot_type, self.filename_prefix+'.' + file_suffix+'.squared_error.'+self.file_format)
        vis.plot_performance(parameters, score_val, score_train, 'score', \
            x_label, plot_type, self.filename_prefix+'.' + file_suffix+'.score.'+self.file_format)

#---------------------------------------------------------------------    
    def write_output(self,performance,input_var,file_suffix):
        '''
        For a given data set (eg: training, validation) write the regression 
        performance measures to file for the associated input parameters
        (eg: regularisation level).
        '''
        filename = self.filename_prefix + '.' + file_suffix + '.dat'
        file = open(filename,'w')
        file.write('#\n')
        file.write('# {0}, squared error, score\n'.format(input_var))
        file.write('#\n')
        for parameter, score, error in performance:
            file.write('%18.10e %18.10e %18.10e\n' % (parameter, error, score))
        file.close()
        return
        
#=====================================================================
# EOF
#=====================================================================

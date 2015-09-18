#=====================================================================
# classification.py
'''
Contains class ClassificationAlgorythm which inherents its properties 
from base class SupervisedMachineLearningAlgorythm.
'''

import numpy as np
import sys

import sklearn as skl
from sklearn.svm import SVC

from libMachineLearning.machine_learning import SupervisedMachineLearningAlgorythm
import libVisualisation.visualisation as vis


#=====================================================================
class ClassificationAlgorythm(SupervisedMachineLearningAlgorythm):
    '''
    Class to build and test classification machine learning algorythms.
    '''
    
#---------------------------------------------------------------------        
    def __init__(self, Xpos, Xneg, ypos, yneg, var_name, classification_type, file_format, \
        number_of_regularisation_levels, min_log_of_regularisation_level, max_log_of_regularisation_level, \
        number_of_proportions_of_samples, min_proportion_of_samples, max_proportion_of_samples=1.0, \
        tolerance=1e-3):
        '''
        Initiates input data of positive (Xpos) and negaitive (Xneg) events
        and output positive (ypos) and negative (yneg) events.
        Classification model is assigned and master class SupervisedMachineLearningAlgorythm
        initiated.
        '''
        self.Xpos = Xpos
        self.Xneg = Xneg
        self.ypos = ypos
        self.yneg = yneg
        self.Xpos, self.ypos = self.shuffle_matrices(np.vstack((self.Xpos, ypos)))
        self.Xneg, self.yneg = self.shuffle_matrices(np.vstack((self.Xneg, yneg)))

        if (classification_type=="logistic_l1"):
            print("\n\nPerforming Logistic Classification on variable {0} using L1 regularisation".format(var_name))
            self.model = skl.linear_model.LogisticRegression(penalty='l1',tol=tolerance)
            self.filename_prefix = 'classification.'+var_name+'.l1.logistic'
        elif (classification_type=="logistic_l2"):
            print("\n\nPerforming Logistic Classification on variable {0} using L2 regularisation".format(var_name))
            self.model = skl.linear_model.LogisticRegression(penalty='l2',tol=tolerance)
            self.filename_prefix = 'classification.'+var_name+'.l2.logistic'
        elif (classification_type=="svm_rbf"):
            print("\n\nPerforming Support Vector Machine Classification of Guassian basis function on variable {0}".format(var_name))
            self.model = SVC(kernel='rbf',tol=tolerance)
            self.filename_prefix = 'classification.'+var_name+'.svm_rbf'
        elif (classification_type=="svm_degree01"):
            print("\n\nPerforming Support Vector Machine Classification of degree 1 on variable {0}".format(var_name))
            self.model = SVC(kernel='poly',tol=tolerance,degree=1)
            self.filename_prefix = 'classification.'+var_name+'.svm_degree01'  
        else:
            print("\n\nClassification type not valid: logistic_l1; logistic_l2; svm_rbf; svm_degree01")
            sys.exit()  
            
        SupervisedMachineLearningAlgorythm.__init__(self, var_name, file_format,\
            number_of_regularisation_levels, min_log_of_regularisation_level, max_log_of_regularisation_level, \
            number_of_proportions_of_samples, min_proportion_of_samples, max_proportion_of_samples) 

#---------------------------------------------------------------------    
    def split_data(self, p0=1.0, print_output=False):
        '''
        Splits data into training, validation and testing sets.
        '''
        if (print_output):
            print("      Splitting positive classification samples ...")
        num_samples_pos, num_train_samples_pos, num_val_samples_pos, num_test_samples_pos, \
            X_train_pos, X_val_pos, X_test_pos, y_train_pos, y_val_pos, y_test_pos \
            = self.split_this_data(self.Xpos, self.ypos, p0, print_output)

        if (print_output):
            print("      Splitting negative classification samples ...")
        num_samples_neg, num_train_samples_neg, num_val_samples_neg, num_test_samples_neg, \
            X_train_neg, X_val_neg, X_test_neg, y_train_neg, y_val_neg, y_test_neg \
            = self.split_this_data(self.Xneg, self.yneg, p0, print_output)

        num_samples	      = num_samples_pos	    + num_samples_neg
        num_train_samples   = num_train_samples_pos + num_train_samples_neg
        num_val_samples     = num_val_samples_pos   + num_val_samples_neg
        num_test_samples    = num_test_samples_pos  + num_test_samples_neg

        X_train = np.hstack((X_train_pos,X_train_neg))
        X_val   = np.hstack((X_val_pos,  X_val_neg))
        X_test  = np.hstack((X_test_pos, X_test_neg))

        y_train = np.hstack((y_train_pos,y_train_neg))
        y_val   = np.hstack((y_val_pos,  y_val_neg))
        y_test  = np.hstack((y_test_pos, y_test_neg))

        return num_samples, num_train_samples, num_val_samples, num_test_samples, \
            X_train, X_val, X_test, y_train, y_val, y_test

#---------------------------------------------------------------------    
    def build_model(self, X0, y0, regularisation_level):
        '''
        Assigns regularisation level and builds model linking input data (X0)
        to output data (y0).
        '''
        self.model.set_params(C=regularisation_level).fit(X0.T, y0)

#---------------------------------------------------------------------            
    def evaluate_model_performance(self, X0, y0, parameter, print_output=False, plot_output=False, file_prefix=''):
        '''
        Evaluates the score, precision, recall and F1 score of the classification prediction,
        and returns as a tuple including the input parameter.
        For regularisation studies, the input parameter is the regularisation level.
        For learning curves, the input parameter is the proportion of total sampels.
        '''
        score = self.model.score(X0.T,y0)
        y_predict = self.model.predict(X0.T)
        
        P = 0						# precision  = True positive / (True positive + False positive)
        if (np.sum(y_predict)>0):
            P = np.sum(y_predict*y0) / np.sum(y_predict)
        R = 0						# recall     = True positive / (True positive + False negative)
        if (np.sum(y0)>0):
            R = np.sum(y_predict*y0) / np.sum(y0)
        F1 = 0						# F1 score
        if (P+R>0):
            F1 = 2.0*P*R/(P+R)
        if (print_output):
            print("      num positive = {0}".format(np.sum(y0)))
            print("      num predicted positive = {0}".format(np.sum(y_predict)))
            print("      num correctly predicted = {0}".format(np.sum(y_predict*y0)))
            print("      precision = {0}".format(P))
            print("      recall = {0}".format(R))
            print("      F1 score = {0}".format(F1))
        return (parameter, score, P, R, F1)

#---------------------------------------------------------------------            
    def calculate_best_regularisation_level(self):
        '''
        The best regularisation level for classification studies is defined
        as that with the maximum F1 score.
        '''
        max_F1 = 0.0
        best_regularisation_level = 0.0
        for regularisation_level, score, P, R, F1 in self.validation_performance_regularisation:
            if (F1>max_F1):
                max_F1 = F1
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
        P_train = []
        R_train = []
        F1_train = []
        for parameter, score, P, R, F1 in training_performance:
            parameters.append(parameter)
            score_train.append(score)
            P_train.append(P)
            R_train.append(R)
            F1_train.append(F1)

        score_val = []
        P_val = []
        R_val = []
        F1_val = []
        for parameter, score, P, R, F1 in validation_performance:
            score_val.append(score)
            P_val.append(P)
            R_val.append(R)
            F1_val.append(F1)
            
        vis.plot_performance(parameters, P_val, P_train, 'precision', \
            x_label, plot_type, self.filename_prefix+'.' + file_suffix+'.precision.'+self.file_format)
        vis.plot_performance(parameters, R_val, R_train, 'recall', \
            x_label, plot_type, self.filename_prefix+'.' + file_suffix+'.recall.'+self.file_format)
        vis.plot_performance(parameters, F1_val, F1_train, 'F1 score', \
            x_label, plot_type, self.filename_prefix+'.' + file_suffix+'.F1_score.'+self.file_format)
        vis.plot_performance(parameters, score_val, score_train, 'score', \
            x_label, plot_type, self.filename_prefix+'.' + file_suffix+'.score.'+self.file_format)

#---------------------------------------------------------------------    
    def write_output(self,performance,input_var,file_suffix):
        '''
        For a given data set (eg: training, validation) write the classification 
        performance measures to file for the associated input parameters
        (eg: regularisation level).
        '''
        filename = self.filename_prefix + '.' + file_suffix+'.dat'
        file = open(filename,'w')
        file.write('#\n')
        file.write('# {0}, precision, recall, F1 score, score\n'.format(input_var))
        file.write('#\n')
        for parameter, score, P, R, F1 in performance:
            file.write('%18.10e %18.10e %18.10e %18.10e %18.10e\n' % (parameter, P, R, F1, score))
        file.close()
        return
        
#=====================================================================
# EOF
#=====================================================================

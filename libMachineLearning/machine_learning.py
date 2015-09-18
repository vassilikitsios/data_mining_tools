#=====================================================================
# machine_learning.py  containing class SupervisedMachineLearningAlgorythm
import numpy as np

#=====================================================================
class SupervisedMachineLearningAlgorythm(object):
    '''
    Base class with common code to build and test both RegressionAlgorythm and 
    ClassificationAlgorythm classes.
    '''

#---------------------------------------------------------------------        
    def __init__(self, var_name, file_format,\
        number_of_regularisation_levels, min_log_of_regularisation_level, max_log_of_regularisation_level,
        number_of_proportions_of_samples, min_proportion_of_samples, max_proportion_of_samples=1.0):
        '''
        Initiates the input variables for the base class.
        '''
        self.var_name = var_name
        self.file_format = file_format
        
        self.number_of_regularisation_levels = number_of_regularisation_levels
        self.min_log_of_regularisation_level = min_log_of_regularisation_level
        self.max_log_of_regularisation_level = max_log_of_regularisation_level
        self.regularisation_levels = np.logspace(self.min_log_of_regularisation_level, \
            self.max_log_of_regularisation_level, self.number_of_regularisation_levels)
            
        self.training_performance_regularisation = []
        self.validation_performance_regularisation = []
        
        self.number_of_proportions_of_samples = number_of_proportions_of_samples
        self.min_proportion_of_samples = min_proportion_of_samples
        self.max_proportion_of_samples = max_proportion_of_samples
        self.proportion_of_samples = np.linspace(self.min_proportion_of_samples, \
            self.max_proportion_of_samples, self.number_of_proportions_of_samples)   
            
        self.training_performance_learning_curves = []
        self.validation_performance_learning_curves = []

        return

#---------------------------------------------------------------------    
    def shuffle_matrices(self,Z):
        '''
        Shuffles order of samples out inputs (X) and outputs (y).
        '''
        np.random.shuffle(Z.T)
        X0 = Z[:-1,:]
        y0 = Z[-1,:]
        return X0,y0

#---------------------------------------------------------------------    
    def split_this_data(self, X0, y0, proporition_of_data=1.0, print_output=False):
        '''
        Splits data into training, validation and testing sets.
        '''
        num_samples = len(y0)                                   # number of all samples
        N = int(num_samples*0.6)                                # maximum number of training samples
        num_train_samples = int(N*proporition_of_data)          # actual number of training samples
        num_val_samples = int(num_samples*0.2)                  # number of validation samples
        num_test_samples = num_samples - N - num_val_samples    # number of test samples

        X_train	= X0[:,0			:num_train_samples]
        X_val	= X0[:,N			:N+num_val_samples]
        X_test	= X0[:,N+num_val_samples	:num_samples]
        
        y_train	= y0[  0			:num_train_samples]
        y_val	= y0[  N			:N+num_val_samples]
        y_test	= y0[  N+num_val_samples	:num_samples]

        if (print_output):
            print("         number of total      samples            = {0}".format(num_samples))
            print("         number of training   samples (max)      = {0}".format(N))
            print("         number of training   samples (selected) = {0}".format(num_train_samples))
            print("         number of validation samples            = {0}".format(num_val_samples))
            print("         number of test       samples            = {0}".format(num_test_samples))

        return num_samples, num_train_samples, num_val_samples, num_test_samples, \
            X_train, X_val, X_test, y_train, y_val, y_test
        
#---------------------------------------------------------------------    
    def split_data(self, proporition_of_data=1.0, print_output=False):
        '''
        Specific implementations in ClassificationAlgorythm and 
        RegressionAlgorythm subclasses.
        '''
        raise NotImplementedError

#---------------------------------------------------------------------    
    def build_model(self, X0, y0, parameter):
        '''
        Specific implementations in ClassificationAlgorythm and 
        RegressionAlgorythm subclasses.
        '''
        raise NotImplementedError

#---------------------------------------------------------------------            
    def evaluate_model_performance(self, y_predict, y0, print_output=False, plot_output=False, file_prefix=''):
        '''
        Specific implementations in ClassificationAlgorythm and 
        RegressionAlgorythm subclasses.
        '''
        raise NotImplementedError

#---------------------------------------------------------------------    
    def regularise_model(self):
        '''
        Calculating regularisation curves on training and cross validation data sets.
        '''
        print("\n   Calculating regularisation curves on training and cross validation data sets ...")
        num_samples, num_train_samples, num_val_samples, num_test_samples, \
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        for regularisation_level in self.regularisation_levels:
            print("         regularisation parameter = {0}".format(regularisation_level))
            self.build_model(X_train, y_train, regularisation_level)
            self.training_performance_regularisation.append(self.evaluate_model_performance(X_train,y_train,regularisation_level))
            self.validation_performance_regularisation.append(self.evaluate_model_performance(X_val,y_val,regularisation_level))
            
        self.plot_output(self.validation_performance_regularisation,\
            self.training_performance_regularisation,'regularisation level','logX',\
            'regularisation')
        self.write_output(self.training_performance_regularisation,\
            'regularisation level','regularisation.training')
        self.write_output(self.validation_performance_regularisation,\
            'regularisation level','regularisation.validation')
        
        best_regularisation_level = self.calculate_best_regularisation_level()
        print("      regularisation value  of best fit = {0}".format(best_regularisation_level))
        return best_regularisation_level

#---------------------------------------------------------------------    
    def calculate_best_regularisation_level(self):
        '''
        Specific implementations in ClassificationAlgorythm and 
        RegressionAlgorythm subclasses.
        '''
        raise NotImplementedError
        
#---------------------------------------------------------------------    
    def calculate_learning_curves(self, regularisation_level):
        '''
        Calculates the learning curves on cross validation data set.
        '''
        print("\n   Calculating learning curves on cross validation data set with regularisation level {0} ...".format(regularisation_level))
        for p in self.proportion_of_samples:
            num_samples, num_train_samples, num_val_samples, num_test_samples, \
                X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(p)
            self.build_model(X_train, y_train, regularisation_level)
            self.training_performance_learning_curves.append(self.evaluate_model_performance(X_train,y_train,p))
            self.validation_performance_learning_curves.append(self.evaluate_model_performance(X_val,y_val,p))            
            del  X_train, X_val, X_test, y_train, y_val, y_test
            
        self.plot_output(self.validation_performance_learning_curves,\
            self.training_performance_learning_curves,'proportion of total training samples','linear',\
            'learning_curves')
        self.write_output(self.training_performance_learning_curves,\
            'proportion of samples','learning_curves.training')
        self.write_output(self.validation_performance_learning_curves,\
            'proportion of samples','learning_curves.validation')
        return

#---------------------------------------------------------------------    
    def plot_output(self,validation_performance,training_performance,x_label,plot_type,file_suffix):
        '''
        Specific implementations in ClassificationAlgorythm and 
        RegressionAlgorythm subclasses.
        '''
        raise NotImplementedError

#---------------------------------------------------------------------    
    def write_output(self,performance,input_var,file_suffix):
        '''
        Specific implementations in ClassificationAlgorythm and 
        RegressionAlgorythm subclasses.
        '''
        raise NotImplementedError
        
#=====================================================================
# EOF
#=====================================================================

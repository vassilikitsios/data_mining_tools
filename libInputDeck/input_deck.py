#=====================================================================
# input_deck.py containing class InputDeck
import string
import sys

#=====================================================================
class InputDeck(object):
    '''
    Class used to read in data from input deck. 
    '''

#---------------------------------------------------------------------    
    def __init__(self):
        '''
        Constructor that:
            1) initialises the dictionary containing the required input
            values associated with the correct type;
            2) and initialises the dictionary containing that will contain
            the actual values read in from the input deck file.
        '''
        self.__STRING = 0
        self.__INTEGER = 1
        self.__REAL = 2
        self.__BOOLEAN = 3
        self.__ValidConfigVariables = {\
            "number_of_regularisation_levels_regression" : self.__INTEGER,
            "min_log_of_regularisation_level_regression" : self.__REAL,
            "max_log_of_regularisation_level_regression" : self.__REAL,
            "number_of_proportions_of_samples_regression" : self.__INTEGER,
            "min_proportion_of_samples_regression" : self.__REAL,
            "number_of_regularisation_levels_classification" : self.__INTEGER,
            "min_log_of_regularisation_level_classification" : self.__REAL,
            "max_log_of_regularisation_level_classification" : self.__REAL,
            "number_of_proportions_of_samples_classification" : self.__INTEGER,
            "min_proportion_of_samples_classification" : self.__REAL,
            "data_dir" : self.__STRING,
            "database" : self.__STRING,
            "perform_classification" : self.__BOOLEAN,
            "shuffle_input_data" : self.__BOOLEAN,
            "perform_regression" : self.__BOOLEAN,
            "perform_visualisation" : self.__BOOLEAN,
            "file_format" : self.__STRING,
            }
        self.config = {}
        return

#---------------------------------------------------------------------
    def read(self,filename):
        '''
        Reads in input parameters from file.
        '''
        with open(filename, "rt") as f:
            for line in f.readlines():
                # Check and remove comment lines
                li = line.strip()
                if li[0] == "#":
                    continue

                items = li.split("=")
                if len(items) != 2:
                    print("Invalid Line: '{0}'".format(li))
                    sys.exit(1)

                key = items[0].strip()
                value = items[1].strip()
                
                if key not in self.__ValidConfigVariables:
                    print("Invalid config variable: '{0}'".format(key))
                    sys.exit(1)

                if self.__ValidConfigVariables[key] == self.__STRING:
                    self.config.update({key: value})
                elif self.__ValidConfigVariables[key] == self.__INTEGER:
                    self.config.update({key: int(value)})
                elif self.__ValidConfigVariables[key] == self.__REAL:
                    self.config.update({key: float(value)})
                elif self.__ValidConfigVariables[key] == self.__BOOLEAN:
                    if value.lower() == "true":
                        self.config.update({key: True})
                    elif value.lower() == "false":
                        self.config.update({key: False})
                    else:
                        print("Variable '{0}' should be true or false not '{1}'".format(key, value))
                        
        # Still need to add to code ot check that all required parameters are specified
        return

#=====================================================================
# EOF
#=====================================================================

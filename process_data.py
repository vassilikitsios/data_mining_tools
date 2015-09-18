# #!/usr/bin/env python
#=====================================================================
import sys
import pandas as pd

import libClean.clean as cl
import libClean.clean_sql as cl_sql
from libInputDeck.input_deck import InputDeck
from libMachineLearning.classification import ClassificationAlgorythm 
from libMachineLearning.regression import RegressionAlgorythm 

#=====================================================================
# This code is used to generate results for various blog posts hosted at 
#       http://dataminingtheworld.blogspot.com.au/
# including:
#	linear regression of life expectancy with regularisation and learning curves
# 	classification of OECD countries with regularisation and learning curves

#=====================================================================
# Main program
#=====================================================================
if __name__ == '__main__':
    print('\n===================================================')
    print('Running '.format(sys.argv[0]))
    print('===================================================\n')

    #-------------------------------------------------------------
    print('Reading input deck ...')
    if (len(sys.argv[:])!=2):
        print('ERROR. Correct usage: {0} <input_deck_file_name>')
        sys.exit()
    input_deck = InputDeck()
    input_deck.read(sys.argv[1].strip())
    
    #-------------------------------------------------------------
    print('\nReading data ...')
    if (input_deck.config['database'] == "pandas"):
        # read data, drop redundant fields and save in pandas database
        df  = cl.read_and_clean_data(input_deck.config['data_dir'])
    elif (input_deck.config['database'] == "SQL"):
        # read data and drop redundant fields and save in sql database
        df  = cl_sql.read_and_clean_data(input_deck.config['data_dir'])
    else:
        print("database type '{0}' not supported".format(input_deck.config['database']))
        print("      try: 'pandas' or 'SQL'")
        sys.exit()

    #-------------------------------------------------------------
    df2 = (df - df.mean()) / df.std()       # standardise
    df2 = df2.dropna()                      # drop missing values

    #-------------------------------------------------------------
    if (input_deck.config['perform_regression']):
        X,y = cl.set_up_regression_matrices(df2)        
        lasso = RegressionAlgorythm(X,y,"lifeExp","lasso",\
            input_deck.config['file_format'],\
            input_deck.config['number_of_regularisation_levels_regression'], \
            input_deck.config['min_log_of_regularisation_level_regression'], \
            input_deck.config['max_log_of_regularisation_level_regression'], \
            input_deck.config['number_of_proportions_of_samples_regression'], \
            input_deck.config['min_proportion_of_samples_regression'])
        best_regression_regulariation_level = lasso.regularise_model()
        lasso.calculate_learning_curves(best_regression_regulariation_level)        
        del X, y
        
    #-------------------------------------------------------------
    if (input_deck.config['perform_classification']) or (input_deck.config['perform_visualisation']):
        if (input_deck.config['database'] == "pandas"):
            df_oecd = pd.read_csv(input_deck.config['data_dir']+'/classifier_oecd.csv', header=0,index_col=0)
            df_oecd = df_oecd.T.ffill().T			# forward fill missing values
        elif (input_deck.config['database'] == "SQL"):
            df_oecd = cl_sql.read_field(input_deck.config['data_dir'], 'oecd', 'oecd')
            df_oecd = df_oecd.ffill()				# forward fill missing values
        df_oecd.fillna(0,inplace=True)
        df_oecd_s = pd.DataFrame(df_oecd.stack().values, columns=['oecd'], index=df_oecd.stack().index)
        # add oecd classification field
        df2 = pd.merge(df2,df_oecd_s,left_index=True,right_index=True,how='outer')	
        # back & forward fill missing values
        df2 = df2.unstack(0).bfill().ffill().stack()					
        df2 = df2.dropna()       
    
    #-------------------------------------------------------------
    if (input_deck.config['perform_classification']):
        Xpos,Xneg,ypos,yneg = cl.set_up_classification_matrices(df2)
        logistic_l1 = ClassificationAlgorythm(Xpos,Xneg,ypos,yneg,"oecd","logistic_l1",\
            input_deck.config['file_format'],
            input_deck.config['number_of_regularisation_levels_classification'], \
            input_deck.config['min_log_of_regularisation_level_classification'], \
            input_deck.config['max_log_of_regularisation_level_classification'], \
            input_deck.config['number_of_proportions_of_samples_classification'], \
            input_deck.config['min_proportion_of_samples_classification'])
        best_classification_regulariation_level = logistic_l1.regularise_model()
        logistic_l1.calculate_learning_curves(best_classification_regulariation_level)
        del Xpos, Xneg, ypos, yneg

    #-------------------------------------------------------------
    if (input_deck.config['perform_visualisation']):
        print('\nVisualising the results ...\n')
        df3 = df2.copy()
        df3.drop('death', axis=1, inplace=True)
        df3.drop('gdpPC', axis=1, inplace=True)
        df3.drop('healthPCGov', axis=1, inplace=True)
        df3.drop('menSchool', axis=1, inplace=True)
        df3.drop('womenSchool', axis=1, inplace=True)
        df3.drop('pop', axis=1, inplace=True)
        df3.drop('popGrowth', axis=1, inplace=True)
        df3.drop('immigration', axis=1, inplace=True)
        
        import libVisualisation.visualisation_sb as vis_sb
        vis_sb.plot_pair(df2, 'pairplot.all.'+input_deck.config['file_format'])
        vis_sb.plot_pair(df3, 'pairplot.'+input_deck.config['file_format'])
        vis_sb.plot_correlations(df2, False, 'corrplot.all.'+input_deck.config['file_format'])
        vis_sb.plot_correlations(df3, True, 'corrplot.'+input_deck.config['file_format'])
        
        df2.drop('oecd', axis=1, inplace=True)
        df3.drop('oecd', axis=1, inplace=True)
        vis_sb.plot_correlations(df2, False, 'corrplot.all_no_OECD.'+input_deck.config['file_format'])
        vis_sb.plot_correlations(df3, True, 'corrplot.no_OECD.'+input_deck.config['file_format'])
        vis_sb.plot_pair(df2, 'pairplot.all_no_OECD.'+input_deck.config['file_format'])
        vis_sb.plot_pair(df3, 'pairplot.no_OECD.'+input_deck.config['file_format'])
        
    print ('\n===================================================')
    print ('Code complete.')
    print ('===================================================\n')

#=====================================================================
# EOF
#=====================================================================

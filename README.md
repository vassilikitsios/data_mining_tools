#Python Data Mining Tools:

1) Pandas is used for general data wrangling.

2) Data can be read in directly from a text file into the pandas data frame, or via an SQL database.

3) Regression and Classifcation machine learning tasks are performed using scikit-learn.

#The files and directories included in this package are as follows:
	process_data.in			- exmaple input deck file containing the input parameters
	process_data.py			- main program
	README.md			- this read me file
	run				- run script to run the main program

	libClean/			- library containing the subroutines undertaking the data reading and cleaning
		clean.py		- source to read data directly into a pandas dataframe
		clean_sql.py		- source to read data into SQL database

	libInputDeck/			- library containing source to process the input deck
		input_deck.py		- source to process the input deck

	libMachineLearning/		- library containing the supervised machine learning tools
		classification.py	- classification specific class
		machine_learning.py	- general base class
		regression.py		- regression specific class

	libVisualisation/		- library containing the subroutines for visualisation the output
		visualisation.py	- source to produce standard matplotlib line plots
		visualisation_sb.py	- source to produce seaborn correlation and pair plots

#To do list

1) add pandas and SQL to a master database class with additional options for: MongoDB; Hadoop; Spark

2) integrate theano neural network source into machine learning library

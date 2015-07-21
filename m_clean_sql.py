#=====================================================================
# import libraries
import sys
import pandas as pd
import numpy as np
import scipy as sci
import string
import sklearn as skl

import mysql.connector
from mysql.connector import errorcode
import pandas.io.sql as pdsql

import m_machine_learning as ml
import m_clean as cl

#=====================================================================
def read_field(input_dir, database_name, table_name):
	config = {'user': 'vassili', 'password': 'vk_mysql_pass', 'host': 'localhost', 'raise_on_warnings': True, 'allow_local_infile': True}
	cnx = mysql.connector.connect(**config)
	cur = cnx.cursor()
	cur.execute('use {0};'.format(database_name))
	df = pdsql.read_frame('select * from oecd.{0}'.format(table_name), cnx)
	#df = pdsql.read_sql('select * from oecd.{0}'.format(var_name), cnx)
	df.index = df['country']
	del df['country']
	return df.T

#=====================================================================
def read_and_clean_data(input_dir):

	#-------------------------------------------------------------
	# list of input data properties

	input_data_properties = list()
	# deaths per 1000 population per year
	input_data_properties.append(('death',input_dir+'/indicator_crude_death_rate__deaths_per_1000_population.csv','double'))
	# total births per year
	input_data_properties.append(('birth',input_dir+'/indicator_estimated_new_births.csv','double'))
	# deaths of children under 5 per 1000 population per year
	input_data_properties.append(('deathU5',input_dir+'/indicator_gapminder_under5mortality.csv','double'))
	# gdp per capita PPP (purching power parity) - check website for more details
	input_data_properties.append(('gdpPC',input_dir+'/indicator_gdppercapita_with_projections.csv','double'))
	# Per capita total expenditure on health at average exchange rate (US$)
	input_data_properties.append(('healthPC',input_dir+'/indicator_health_spending_per_person_US.csv','double'))
	# average life expectancy at birth - check website for more details 
	input_data_properties.append(('lifeExp',input_dir+'/indicator_life_expectancy_at_birth_1800-2050.csv','double'))
	# Per capita general government expenditure on health expressed at average exchange rate for that year in US dollars
	input_data_properties.append(('healthPCGov',input_dir+'/indicator_per_capita_government_expenditure_on_health_at_average_exchange_rate_US.csv','double'))
	# Population growth (annual %). Derived from total population.
	input_data_properties.append(('popGrowth',input_dir+'/population_growth.csv','double'))
	# The average number of years of school (primary, secondary, tertiary) attended by all people in the age and gender group specified.
	input_data_properties.append(('womenSchool',input_dir+'/Years_in_school_women_25_plus.csv','double'))
	input_data_properties.append(('menSchool',input_dir+'/Years_in_school_men_25_plus.csv','double'))
	# total population
	input_data_properties.append(('pop',input_dir+'/indicator_gapminder_population.csv','double'))
	# the year that the country entres into the oecd
#	input_data_properties.append(('oecd',input_dir+'/classifier_oecd.csv','double'))

	#-------------------------------------------------------------
	config = {'user': 'vassili', 'password': 'vk_mysql_pass', 'host': 'localhost', 'raise_on_warnings': True, 'allow_local_infile': True}
	cnx = mysql.connector.connect(**config)
	cur = cnx.cursor()
	
	#-------------------------------------------------------------
	# create database
	database_name = 'oecd'
	#create_sql_database(cur,database_name)
	cur.execute('use {0};'.format(database_name))

	#-------------------------------------------------------------
	# create mySQL tables and populate with data from file
	for i in range(0,len(input_data_properties)):
		var_name,filename,var_type = input_data_properties[i]
		#populate_sql_table(cur,var_name,filename,var_type)

	#-------------------------------------------------------------
	# read from mySQL into pandas data frame, then merge and align
	for i in range(0,len(input_data_properties)):
		var_name,filename,var_type = input_data_properties[i]
		this_df = pdsql.read_frame('select * from oecd.{0}'.format(var_name), cnx)
		#this_df = pdsql.read_sql('select * from oecd.{0}'.format(var_name), cnx)
		this_df.index = this_df['country']
		del this_df['country']
		this_df_stacked = pd.DataFrame(this_df.T.stack().values,columns=[var_name],index=this_df.T.stack().index)
		if (i==0):
			df = this_df_stacked.copy()
		else:
        		df = pd.merge(df,this_df_stacked.copy(),left_index=True,right_index=True,how='outer')
		del this_df, this_df_stacked
	cnx.close()

	#-------------------------------------------------------------
	df = cl.scale_variables(df)

	return df

#=====================================================================
def create_sql_database(cur,database_name):
	try:
		print("Creating database {0}: ".format(database_name))
		cur.execute('create database if not exists {0};'.format(database_name))
	except mysql.connector.Error as err:
		if err.errno == 1007:
			print("      already exists.\n")
			cur.execute("drop database {0};".format(database_name));
			cur.execute('create database if not exists {0};'.format(database_name))
		else:
			print(err.msg)
	else:
		print("      OK\n")
	return

#=====================================================================
def create_sql_table(cur,table_name):
	try:
		print("Creating table {0}: ".format(table_name))
		cur.execute("create table if not exists `{0}` ("
			"`country`      VARCHAR(200),"
			"PRIMARY KEY    (`country`)"
			");".format(table_name))
	except mysql.connector.Error as err:
		if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
			print("      already exists.\n")
			cur.execute("drop table {0};".format(table_name));
			cur.execute("create table if not exists `{0}` (`country` VARCHAR(200), PRIMARY KEY (`country`));".format(table_name))
		else:
			print(err.msg)
	else:
		print("      OK\n")
	return

#=====================================================================
def get_column_names(filename):
	fin = open(filename,'r')
	linestring = fin.readline()
	column_names = linestring.rstrip('\n').split(",")
	fin.close()
	return column_names

#=====================================================================
def populate_sql_table(cur,table_name,filename,var_type):
	create_sql_table(cur,table_name)
	column_names = get_column_names(filename)
	for i in range(1,len(column_names)):
		cur.execute('alter table `{0}` add column `{1}` {2} default null;'.format(table_name, column_names[i], var_type))
	cur.execute('load data local infile "{0}" into table {1} columns terminated by "," lines terminated by "\\n" ignore 1 lines;'.format(filename,table_name))
	return

#=====================================================================
# EOF
#=====================================================================

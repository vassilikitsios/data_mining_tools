#=====================================================================
# import libraries
import sys
import pandas as pd
import numpy as np
import scipy as sci
import sklearn as skl

import m_machine_learning as ml

#=====================================================================
def read_and_clean_data():
	# deaths per 1000 population per year
	df_death=pd.read_csv('./0data/indicator_crude_death_rate__deaths_per_1000_population.csv', header=0,index_col=0)
	# total births per year
	df_birth=pd.read_csv('./0data/indicator_estimated_new_births.csv', header=0,index_col=0)
	# deaths of children under 5 per 1000 population per year
	df_deathU5=pd.read_csv('./0data/indicator_gapminder_under5mortality.csv', header=0,index_col=0)
	# gdp per capita PPP (purching power parity) - check website for more details
	df_gdpPC=pd.read_csv('./0data/indicator_gdppercapita_with_projections.csv', header=0,index_col=0)
	# Per capita total expenditure on health at average exchange rate (US$)
	df_healthPC=pd.read_csv('./0data/indicator_health_spending_per_person_US.csv', header=0,index_col=0)
	# average life expectancy at birth - check website for more details 
	df_lifeExp=pd.read_csv('./0data/indicator_life_expectancy_at_birth_1800-2050.csv', header=0,index_col=0)
	# Per capita general government expenditure on health expressed at average exchange rate for that year in US dollars
	df_healthPCGov=pd.read_csv('./0data/indicator_per_capita_government_expenditure_on_health_at_average_exchange_rate_US.csv', header=0,index_col=0)
	# Population growth (annual %). Derived from total population.
	# Annual population growth rate for year t is the exponential rate of growth of midyear population from year t-1 to t.
	# Population is based on the de facto definition of population, which counts all residents regardless of legal status or citizenship.
	# Refugees not permanently settled in the country of asylum, are generally considered part of the population of the country of origin.
	df_popGrowth=pd.read_csv('./0data/population_growth.csv', header=0,index_col=0)
	# The average number of years of school (primary, secondary, tertiary) attended by all people in the age and gender group specified.
	df_menSchool=pd.read_csv('./0data/Years_in_school_men_25_plus.csv', header=0,index_col=0)
	df_womenSchool=pd.read_csv('./0data/Years_in_school_women_25_plus.csv', header=0,index_col=0)
	# total population
	df_pop=pd.read_csv('./0data/indicator_gapminder_population.csv', header=0,index_col=0)
	# the year that the country entres into the oecd
	df_oecd=pd.read_csv('./0data/classifier_oecd.csv', header=0,index_col=0)
	df_oecd = df_oecd.T.ffill().T		# forward fill missing values

	#-------------------------------------------------------------
	# Stack
	df_death_s 	= pd.DataFrame(df_death.stack().values,		columns=['death'], 	index=df_death.stack().index)
	df_birth_s 	= pd.DataFrame(df_birth.stack().values, 	columns=['birth'], 	index=df_birth.stack().index)
	df_deathU5_s 	= pd.DataFrame(df_deathU5.stack().values, 	columns=['deathU5'], 	index=df_deathU5.stack().index)
	df_gdpPC_s 	= pd.DataFrame(df_gdpPC.stack().values,	 	columns=['gdpPC'], 	index=df_gdpPC.stack().index)
	df_healthPC_s 	= pd.DataFrame(df_healthPC.stack().values, 	columns=['healthPC'], 	index=df_healthPC.stack().index)
	df_lifeExp_s 	= pd.DataFrame(df_lifeExp.stack().values,	columns=['lifeExp'], 	index=df_lifeExp.stack().index)
	df_healthPCGov_s= pd.DataFrame(df_healthPCGov.stack().values, 	columns=['healthPCGov'],index=df_healthPCGov.stack().index)
	df_popGrowth_s 	= pd.DataFrame(df_popGrowth.stack().values, 	columns=['popGrowth'], 	index=df_popGrowth.stack().index)
	df_womenSchool_s= pd.DataFrame(df_womenSchool.stack().values,	columns=['womenSchool'],index=df_womenSchool.stack().index)
	df_menSchool_s 	= pd.DataFrame(df_menSchool.stack().values, 	columns=['menSchool'], 	index=df_menSchool.stack().index)
	df_pop_s 	= pd.DataFrame(df_pop.stack().values, 		columns=['pop'], 	index=df_pop.stack().index)

	#-------------------------------------------------------------
	# Merge and allign
	df = pd.merge(df_lifeExp_s,	df_death_s,		left_index=True,right_index=True,how='outer')
	df = pd.merge(df,		df_birth_s,		left_index=True,right_index=True,how='outer')
	df = pd.merge(df,		df_deathU5_s,		left_index=True,right_index=True,how='outer')
	df = pd.merge(df,		df_gdpPC_s,		left_index=True,right_index=True,how='outer')
	df = pd.merge(df,		df_healthPC_s,		left_index=True,right_index=True,how='outer')
	df = pd.merge(df,		df_healthPCGov_s,	left_index=True,right_index=True,how='outer')
	df = pd.merge(df,		df_womenSchool_s,	left_index=True,right_index=True,how='outer')
	df = pd.merge(df,		df_menSchool_s,		left_index=True,right_index=True,how='outer')
	df = pd.merge(df,		df_popGrowth_s,		left_index=True,right_index=True,how='outer')
	df = pd.merge(df,		df_pop_s,		left_index=True,right_index=True,how='outer')

	#-------------------------------------------------------------
	# Scale variables to consistent units
	df['popGrowth']	= df['popGrowth'] / 100.0	# change from percentage to proportion	
	df['death']	= df['death'] / 1000.0		# change from deaths per 1000 people to per capita
	df['birth']	= df['birth'] / df['pop']	# change from total births to per capita 
	df['deathU5']	= df['deathU5'] / 1000.0	# change from deaths per 1000 people to per capita

	#-------------------------------------------------------------
	# Outlier detection

	#-------------------------------------------------------------
	# Impute missing values - back and forward fill
	#df = df.unstack(0).bfill().ffill().stack()

	# Impute missing values - replace with median for each variable across all years and countries
	#imputer = skl.preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0, copy=True)
	#df.values[:,:] = imputer.fit_transform(df.values)

	# Impute missing values - replace with median for each variable and country across all years 
	#df = df.unstack(0).fillna(df.unstack(0).median()).stack()

	# Impute missing values - replace with median for each variable and year across all countries 
	#df = df.unstack(1).fillna(df.unstack(1).median()).stack()

	# drop remaining missing values
	#df = df.dropna()

	#-------------------------------------------------------------
	# Calculate derived quantities
	df_immigration_s= pd.DataFrame(df['popGrowth'] - df['birth'] + df['death'], columns=['immigration'], index=df.index)
	df = pd.merge(df,df_immigration_s,left_index=True,right_index=True,how='outer')

	return df

#=====================================================================
def set_up_pair_of_classification_matrices(df2):
	Z = np.vstack(( 
		df2['death'].values, df2['birth'].values, df2['deathU5'].values, df2['gdpPC'].values,\
		df2['healthPC'].values, df2['healthPCGov'].values, df2['womenSchool'].values,\
		df2['menSchool'].values, df2['popGrowth'].values, df2['pop'].values,\
		df2['immigration'].values, df2['lifeExp'].values, df2['oecd'].values))
	X,y = ml.shuffle_matrices(Z)
	return X,y

#=====================================================================
def set_up_classification_matrices(df2):
	Xpos,ypos = set_up_pair_of_classification_matrices(df2[df2['oecd']==1])
	Xneg,yneg = set_up_pair_of_classification_matrices(df2[df2['oecd']==0])
	return Xpos,Xneg,ypos,yneg

#=====================================================================
def set_up_regression_matrices(df2):
	Z = np.vstack(( 
		df2['death'].values, df2['birth'].values, df2['deathU5'].values, df2['gdpPC'].values,\
		df2['healthPC'].values, df2['healthPCGov'].values, df2['womenSchool'].values,\
		df2['menSchool'].values, df2['popGrowth'].values, df2['pop'].values,\
		df2['immigration'].values, df2['lifeExp'].values))
	X,y = ml.shuffle_matrices(Z)
	return X,y

#=====================================================================
# EOF
#=====================================================================

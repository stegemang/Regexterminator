from flask import render_template, request
from app import app

#Import libraries 
import re
import pandas as pd
import numpy as np
import pickle 

import difflib
from sklearn.tree import DecisionTreeClassifier

'''
Load pickled items
'''
#Load pickled model
filename = './app/finalized_model.sav'
clf = pickle.load(open(filename, 'rb'))

#Load pickled column names for transforming input data into dummy format
dummies_cols = pickle.load(open('./app/dummies_cols.pickle', 'rb'))

#Load pickled dictionary of regex values to associate classification output with actual regex pattern
regex_dict = pickle.load(open('./app/regex_dict.pickle', 'rb'))

#Load original format for dummies
#Load dictionary for class

'''
Function to clean-up the input sentences
'''
def cleanInput(a, b):
	"""
	Function to perform feature transformation & engineering 

	@param a: uncleaned sentence
	@param b: desired output sentence 

	@return: feature transformed data
	"""

	d = difflib.Differ()
	diff = d.compare(a,b)
	tmp = list(diff)
	out = pd.DataFrame(data={
	    'sentence': a,
	    'diffs': [i[0] for i in tmp],
	    'char': [i[2] for i in tmp]
	})
	    

	'''
	Using the dummies table from the trained data set, reindex the new dummies to fit the same standard
	https://stackoverflow.com/questions/28465633/easy-way-to-apply-transformation-from-pandas-get-dummies-to-new-data
	'''
	dummies_to_fit = pd.get_dummies(out, columns=['char'])

	dummies_to_fit = dummies_to_fit.reindex(columns = dummies_cols, fill_value=0)

	#Rename the columns
	dummies_to_fit = dummies_to_fit.rename(columns={'char_[': 'char_left_square_bracket', 'char_]': 'char_right_square_bracket',
	                       'char_<': 'char_left_carrot'})

	X_new = dummies_to_fit.drop(['sentence', 'regex','diffs'], axis=1) #Remove everything but the dummy variables

	#For whatever reason it's creating a separate matrix for each one. Add them all together instead
	X_new = X_new.sum().values

	#Reshape data for single sample
	X_new = X_new.reshape(1, -1)

	return X_new

@app.route('/')
@app.route('/index')
def index():
	return render_template("input.html")
	
@app.route('/input')
def cities_input():	#AKA input_sentence
	return render_template("input.html")
	
@app.route('/output')
def cities_output():
	#Input values for the sentences
	s_start = request.args.get('uncleaned_sentence')
	s_end = request.args.get('cleaned_sentence')

	X_new = cleanInput(s_start, s_end) #Clean userInput sentence

	predicted_y = clf.predict(X = X_new) #Use model to predict class
	
	#Match that value back to the initial class
	out = [regex_dict.get(y) for y in predicted_y]

	#Create output for prediction
	return render_template("output.html", y_class = out[0],
		uncleaned_sentence = s_start,
		cleaned_sentence = s_end)
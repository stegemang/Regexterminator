from flask import render_template, request
from app import app

#Import libraries 
import re
import pandas as pd
import numpy as np
import pickle
import string 

import difflib
from sklearn.tree import DecisionTreeClassifier

'''
Load pickled items
'''
#Load pickled model
filename = './app/finalized_model.sav'
clf = pickle.load(open(filename, 'rb'))

#Load pickled dictionary of regex values to associate classification output with actual regex pattern
regex_dict = pickle.load(open('./app/regex_dict.pickle', 'rb'))

#Load original format for dummies
#Load dictionary for class

'''
functiontion to do character counts for cleanup function below
'''
def charCounts(column, colname):
    """
    Counts the different type of characters in a string
    
    @param column: to apply lambda function to
    @param colname: what the unique columns should be named
    
    """
    # This is really ugly, something I could do in R very easily:
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))

    out = column.apply(
    lambda s: pd.Series(
        {colname+'punct': count(s,set(string.punctuation)),
         colname+'letters': count(s,set(string.ascii_letters)),
         colname+'digits': count(s,set(string.digits)),
         colname+'lower': count(s,set(string.ascii_lowercase)),
         colname+'upper': count(s,set(string.ascii_uppercase)),
         colname+'whitespace': count(s,set(string.whitespace)),
         colname+'words': len(s.split()),
        }))
    return out

'''
funciton to evaluate ends of inputs, used by cleanup below
'''

def endChecks(dataf, start_col, end_col, regex = False):
    """
    Checks if end words and characters are the same between two strings (from columns)
    
    @param dataf: pandas dataframe.
    @param start_col: initial string
    @param end_col: modified string to compare to
    @param regex: include regex in new table for indexing.
    
    @return: returns dataframe with start column and regex for indexing, and four new columns of features
    """
    out = pd.DataFrame()
    for index, row in dataf.iterrows():
        tmp = {
            start_col: row[start_col],
            'regex': row['regex'] if regex else '',
            'first_word_same': row[start_col].split()[0] == row[end_col].split()[0] if len(row[end_col].split())>0 else False,
            'first_char_same': row[start_col][0] == row[end_col][0] if len(row[end_col].split())>0 else False,
            'last_word_same': row[start_col].split()[-1] == row[end_col].split()[-1] if len(row[end_col].split())>0 else False,
            'last_char_same': row[start_col][-1] == row[end_col][-1] if len(row[end_col].split())>0 else False,
        }
        out = out.append(tmp, ignore_index=True)
        
    return out

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
    user_inputs = {'sentence': [a], 'end': [b]}
    user_inputs = pd.DataFrame(user_inputs)

    #add in character type counts features

    a_counts = charCounts(user_inputs['sentence'], 'start_n_')
    b_counts = charCounts(user_inputs['end'], 'end_n_')

    input_type_counts = pd.concat([user_inputs, a_counts, b_counts], axis= 1)

    #string ends feature:
    string_ends = endChecks(user_inputs, 'sentence','end', regex = False)

    input_features_combo = pd.merge(input_type_counts, string_ends, on="sentence")

    X_new = input_features_combo.drop(['sentence','regex','end'], axis=1)

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
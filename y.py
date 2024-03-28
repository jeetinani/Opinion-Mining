from flask import Flask,render_template,jsonify,request
import numpy as np
import pickle
import json
import pandas as pd
import numpy as np
import re
import sys
import nltk
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import collections
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
app=Flask(__name__,template_folder='template')
miner=pickle.load(open('model.pkl','rb'))

"""df = pd.read_csv('restaurant.csv')
df = df.rename(columns={'text':'reviews'})
df.head()
corpus=[]
ls=WordNetLemmatizer()
for i in range(0,len(dfs)):
    review=re.sub('[^a-zA-z]',' ',dfs['reviews'][i])
    review=review.lower()
    review=review.split()
    review=[ls.lemmatize(word) for word in review if word not in stopwords or word == 'not']
    review=' '.join(review)
    corpus.append(review)"""


@app.route('/x')
def  home():
	return render_template('x.html')

@app.route('/predict1',methods=['POST'])
def predict1():
	#if request.method == 'POST':
	int_features=request.form['reviews']
	"""corpus.append(int_features)
	tf=TfidfVectorizer(max_features=15000)
	X=tf.fit_transform(corpus).toarray()
	y=dfs.iloc[:,1].values
	tester = np.array([int_features])
	tf=TfidfVectorizer(max_features=15000)
	vector = tf.fit_transform(tester).toarray()
	prediction=miner.predict(vector)"""
	return render_template('x.html',predict_text='{}'.format(int_features))
if __name__ == '__main__':
    app.run(debug=True)
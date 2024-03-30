import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from helpers.Transformer import getTransformedReview

def loadVectorizer():
    return TfidfVectorizer(max_features=15000)

tfidfVectorizer = loadVectorizer()

def loadModel():
    filePath = 'models/model.pkl'
    file=Path(filePath)
    if(file.exists()):
        print ("loading from existing model")
        return pickle.load(open(filePath,'rb'))
    dataSet = pd.read_csv('model-restaurant - restaurant.csv')
    corpus=[]
    for review in dataSet['reviews']:
        corpus.append(getTransformedReview(review))
    X=tfidfVectorizer.fit_transform(corpus).toarray()
    y=dataSet.iloc[:,1].values
    miner=MultinomialNB().fit(X,y)
    print("new model created")
    with open(filePath,'wb') as file:
        pickle.dump(miner,file)
    return miner

miner = loadModel()
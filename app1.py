import numpy as np
import pandas as pd
import re
import pickle
from flask import Flask,render_template,request
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('stopwords')
stopwords = stopwords.words('english')


app=Flask(__name__,template_folder='template',static_folder='static')
miner=pickle.load(open('model.pkl','rb'))

@app.route('/')
def  home_default():
    return render_template('home.html')

@app.route('/home')
def  home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    df = pd.read_csv('model-restaurant.csv')
    #df = df.rename(columns={'text':'reviews'})


    #df['reviews'].isna().sum()
    #df['remove_lower_punct'] = df['reviews']


    analyser = SentimentIntensityAnalyzer()

    """sentiment_score_list = []
    sentiment_label_list = []

    for i in df['remove_lower_punct'].values.tolist():
        sentiment_score = analyser.polarity_scores(i)
        
        if sentiment_score['compound'] >= 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Positive')
        elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Negative')
        elif sentiment_score['compound'] <= -0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Negative')
        
    df['sentiment'] = sentiment_label_list
    df['sentiment score'] = sentiment_score_list"""


    #dfs = df.filter(['reviews','sentiment'], axis=1)


    """data=pd.get_dummies(dfs['sentiment'])
    data=data.drop(['Negative'],axis='columns')
    dfs = dfs.drop(['sentiment'],axis='columns')
    dfs=pd.concat([dfs,data],axis='columns')"""


    corpus=[]
    ls=WordNetLemmatizer()
    for i in range(0,len(df["reviews"])):
        review=re.sub('[^a-zA-z]',' ',df['reviews'][i])
        review=review.lower()
        review=review.split()
        review=[ls.lemmatize(word) for word in review if not word in stopwords or word == 'not']
        review=' '.join(review)
        corpus.append(review)

    tf=TfidfVectorizer(max_features=15000)
    X=tf.fit_transform(corpus).toarray()
    #y=dfs.iloc[:,1].values
        
    #from sklearn.model_selection import train_test_split
    #from sklearn.metrics import confusion_matrix,accuracy_score
    #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    #from sklearn.naive_bayes import MultinomialNB
    #miner=MultinomialNB().fit(X_train,y_train)

    
    int_features=request.form['reviews']
    int_features=re.sub('[^a-zA-z]',' ',int_features)
    int_features=int_features.lower()
    int_features=int_features.split()
    int_features=[ls.lemmatize(word) for word in int_features if word not in stopwords or word == 'not']
    int_features=' '.join(int_features)
    tester = np.array([int_features])
    vector = tf.transform(tester) 
    prediction=miner.predict(vector)
    if(prediction == 0):
        ans="Negative"
    else:
        ans="Positive"

    score_numeric=analyser.polarity_scores(int_features)
    return render_template('home.html',predict_text='review is {}'.format(ans),score='compound score is {}'.format(score_numeric['compound']))
if __name__ == '__main__':
    app.run(debug=True)
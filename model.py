import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from helpers.Words import replace_list, contractions
#nltk.download('stopwords')
stopwords = stopwords.words('english')

""" replace_list=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'can', 'will', 'just', 'now']

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}
 """

df = pd.read_csv('restaurant.csv')
df = df.rename(columns={'text':'reviews'})
#df.head()


df['reviews'].isna().sum()
#df.shape
df['remove_lower_punct'] = df['reviews']



analyser = SentimentIntensityAnalyzer()

sentiment_score_list = []
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
df['sentiment score'] = sentiment_score_list

#display(df.head(10))

dfs = df.filter(['reviews','sentiment'], axis=1)
#dfs.head()


data=pd.get_dummies(dfs['sentiment'])
data=data.drop(['Negative'],axis='columns')
dfs = dfs.drop(['sentiment'],axis='columns')
dfs=pd.concat([dfs,data],axis='columns')
#dfs.head()


corpus=[]
ls=WordNetLemmatizer()
for i in range(0,len(dfs)):
    #review=re.sub('[^a-zA-z]',' ',dfs['reviews'][i])
    review=dfs['reviews'][i].split()
    for j in range(len(review)):
        if(review[j] in contractions.keys()):
            review[j]=contractions[review[j]]
    review=[ls.lemmatize(word) for word in review if not word in replace_list]
    review=' '.join(review)
    corpus.append(review)

tf=TfidfVectorizer(max_features=15000)
X=tf.fit_transform(corpus).toarray()
y=dfs.iloc[:,1].values

l = []


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
miner=MultinomialNB().fit(X_train,y_train)
y_pred=miner.predict(X_test)
print("The Accuracy using Naive Bayes Algorithm is ",accuracy_score(y_test,y_pred)*100)
l.append(accuracy_score(y_test,y_pred)*100)
tester = np.array(['Nothing was good i mean nothing. Hated that place'])
vector = tf.transform(tester)
print(miner.predict(vector))
k = analyser.polarity_scores("Good place. But bad service")
print(k['compound'])


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

miner=MultinomialNB().fit(X_train,y_train)
y_pred=miner.predict(X_test)
file='model.pkl'
with open(file,'wb') as file:
	pickle.dump(miner,file)
#pickle.dump(clf,open('model.pkl','wb'))
#y_pred=clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#print("Accuracy using SVM ",accuracy_score(y_test,y_pred)*100)
#l.append(accuracy_score(y_test,y_pred)*100)


	#d = {'Algorithms':['Naive Bayes','SVM'],'Accuracy':l}
	#finale = pd.DataFrame(d)
	#finale.plot(kind = 'bar',
	 #       x = 'Algorithms',
	  
	  #     y = 'Accuracy',
	   #     color = 'blue')
	#plt.title('BarPlot')
#plt.show()
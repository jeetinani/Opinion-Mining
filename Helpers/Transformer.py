from Words import replace_list, contractions

from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def getTransform(review):
    #df = pd.read_csv('model-restaurant - restaurant.csv')
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
    df['sentiment score'] = sentiment_score_list


    dfs = df.filter(['reviews','sentiment'], axis=1)"""


    """data=pd.get_dummies(dfs['sentiment'])
    data=data.drop(['Negative'],axis='columns')
    dfs = dfs.drop(['sentiment'],axis='columns')
    dfs=pd.concat([dfs,data],axis='columns')"""


    review = review.lower().replace("!"," ").replace(',',' ').replace("."," ").strip().split()
    #replace('[^\w\s]', ' ').str.replace(" \d+", " ").str.replace(' +', ' ').str.strip()

    #df['tokenise'] = df.apply(lambda row: nltk.word_tokenize(row[1]), axis=1)

    #review = review.split()	
    
    #corpus=[]

    wordNetLemmatizer=WordNetLemmatizer()

    temp_review=[]
    for word in review:
        #temp_review=[]
        if(word in contractions.keys()):
            temp_review.append(contractions[word])
        else:
            temp_review.append(word)
    #review=' '.join(temp_review)
    #review=review.split()
    temp_review=[wordNetLemmatizer.lemmatize(word) for word in temp_review if word not in replace_list]
    #df['changed'][i]=' '.join([ls.lemmatize(word) for word in review])
    #temp_review=[ls.lemmatize(word) for word in temp_review]
    review=' '.join(temp_review)
    #df["lemmatise"][i]=review
    #corpus.append(review)
    return review
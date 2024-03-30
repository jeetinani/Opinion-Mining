import numpy as np
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from flask import Flask,render_template,request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from Helpers.Words import replace_list, contractions, getWordsDictionary
#import json
#PEOPLE_FOLDER = os.path.join('static', 'people_photo')

# nltk.download('stopwords')
#stopwords = stopwords.words('english')

""" fw=pd.read_csv("shared words list.csv")
    #print(fw.head())
food=fw["food"].tolist()
for i in range(len(food)):
    food[i]=food[i][1:-1].lower()
    #print(food[:5],len(food))
x=fw["pricing"].tolist()
pricing=[]
for i in range(len(x)):
    if(str(x[i])=="nan"):
        break
    pricing.append(x[i][1:-1].lower())
    #print(pricing[:5],len(pricing))
x=fw["hygiene"].tolist()
hygiene=[]
for i in range(len(x)):
    if(str(x[i])=="nan"):
        break
    hygiene.append(x[i][1:-1].lower())
    #print(hygiene[:5],len(hygiene))
x=fw["service"].tolist()
service=[]
for i in range(len(x)):
    if(str(x[i])=="nan"):
        break
    service.append(x[i][1:-1].lower())
    #print(service[:5],len(service))
x=fw["ambiance"].tolist()
ambiance=[]
for i in range(len(x)):
    if(str(x[i])=="nan"):
        break
    ambiance.append(x[i][1:-1].lower())
    #print(ambiance[:5],len(ambiance))
x=fw["miscelleanous"].tolist()
miscelleanous=[]
for i in range(len(x)):
    if(str(x[i])=="nan"):
        break
    miscelleanous.append(x[i][1:-1].lower())
    #print(miscelleanous[:5],len(miscelleanous))
food=list(set(food))
    #print(food[:5],len(food))
miscelleanous=list(set(miscelleanous))
    #print(miscelleanous[:5],len(miscelleanous))
hygiene=list(set(hygiene))
    #print(hygiene[:5],len(hygiene))
service=list(set(service))
    #print(service[:5],len(service))
pricing=list(set(pricing))
    #print(pricing[:5],len(pricing))
ambiance=list(set(ambiance))
    #print(ambiance[:5],len(ambiance)) """

""" replace_list=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
            'can', 'will', 'just', 'now']


contractions = {"ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because",
    "could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not",
    "don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would",
    "he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you",
    "how'll": "how will","how's": "how is","i'd": "I would","i'd've": "I would have","i'll": "I will","i'll've": "I will have",
    "i'm": "I am","i've": "I have","isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will",
    "it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have",
    "mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
    "needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have",
    "shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she had","she'd've": "she would have",
    "she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have","shouldn't": "should not",
    "shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have",
    "that's": "that is","there'd": "there would","there'd've": "there would have","there's": "there is","they'd": "they would",
    "they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have",
    "to've": "to have","wasn't": "was not","we'd": "we would","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
    "we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
    "what's": "what is","what've": "what have","when's": "when is","when've": "when have","where'd": "where did","where's": "where is",
    "where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is",
    "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
    "wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have",
    "y'all're": "you all are","y'all've": "you all have","you'd": "you would","you'd've": "you would have","you'll": "you will",
    "you'll've": "you will have","you're": "you are","you've": "you have"}
 """

app=Flask(__name__,template_folder='template',static_folder='static')

wordsDictionary = getWordsDictionary()
#app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

#miner=pickle.load(open('model.pkl','rb'))

@app.route('/')
def  home_default():
    return render_template('home.html')

@app.route('/home')
def  home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():



    df = pd.read_csv('model-restaurant - restaurant.csv')
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


    df['remove_lower_punct'] = df['reviews'].str.lower().str.replace("!"," ").str.replace(',',' ').str.replace("."," ").str.strip()#replace('[^\w\s]', ' ').str.replace(" \d+", " ").str.replace(' +', ' ').str.strip()

    #df['tokenise'] = df.apply(lambda row: nltk.word_tokenize(row[1]), axis=1)

    df['tokenise'] = df['remove_lower_punct'].apply(lambda row:row.split())	
    df['changed']= df["tokenise"]
    df['lemmatise'] = df['tokenise']
    
    corpus=[]

    ls=WordNetLemmatizer()

    for i in range(len(df['tokenise'])):
        review=df['tokenise'][i]
        temp_review=[]
        for j in review:
            #temp_review=[]
            if(j in contractions.keys()):
                temp_review.append(contractions[j])
            else:
                temp_review.append(j)
        review=' '.join(temp_review)
        review=review.split()
        review=[word for word in review if word not in replace_list]
        df['changed'][i]=' '.join([ls.lemmatize(word) for word in review])
        review=[ls.lemmatize(word) for word in review]
        review=' '.join(review)
        df["lemmatise"][i]=review
        corpus.append(review)
    #df.to_csv("trial.csv")
    
    """corpus=[]
    ls=WordNetLemmatizer()
    for i in range(0,len(dfs)):
        review=re.sub('[^a-zA-z]',' ',dfs['reviews'][i])
        review=review.lower()
        review=review.split()
        review=[ls.lemmatize(word) for word in review if not word in stopwords or word == 'not']
        review=' '.join(review)
        corpus.append(review)"""

    tf=TfidfVectorizer(max_features=20000)
    X=tf.fit_transform(corpus).toarray()
    y=df.iloc[:,1].values

    """ dfcorp = pd.DataFrame(list(zip(corpus,df.iloc[:,1].values)),columns =['corpus', 'val'])
    dfcorp.to_csv("corp.csv") """    
    
    #from sklearn.model_selection import train_test_split
    #from sklearn.metrics import confusion_matrix,accuracy_score
    #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    #from sklearn.naive_bayes import MultinomialNB
    #miner=MultinomialNB().fit(X_train,y_train)

    original_review=request.form['reviews']
    int_features=original_review
    int_features=int_features.lower()
    #int_features=int_features.replace('[^\w\s]', ' ')
    #int_features=int_features.replace(" \d+", " ")
    #int_features=int_features.replace(' +', ' ').strip()
    int_features=int_features.replace(',',' ').replace("."," ").replace("!"," ").strip()
    #int_features=re.sub('[^a-zA-z]',' ',int_features)
    #int_features=int_features.lower()
    int_features=int_features.split()
    review_tokenised=int_features
    # int_features=[ls.lemmatize(word) for word in int_features if not word in stopwords or word == 'not']
    # int_features=' '.join(int_features)
    temp_review=[]
    for j in int_features:
        if(j in contractions.keys()):
            temp_review.append(contractions[j])
        else:
            temp_review.append(j)
    review=' '.join(temp_review)
    review=review.split()
    review=[word for word in review if word not in replace_list]
    review=[ls.lemmatize(word) for word in review]
    int_features=' '.join(review)


    miner=MultinomialNB().fit(X,y)


    tester = np.array([int_features])
    vector = tf.transform(tester)
    prediction = miner.predict(vector)
    if(prediction == 0):
        ans="Negative"
    else:
        ans="Positive"

    positive_rest = ['good','amazing','awesome','great','tasty','divine','delicious','yumm','yummy','fingerlicking',
        'heavenly','appetizing','flavorful','palatable','flavorful','flavorsome','good-tasting','superb',
        'fingerlicking','flawless','beautiful',"garnishing","garnished","sweet","savoury",'enjoy','enjoyed']
    negative_rest = ['old','yuck','ewww','expensive','costly','inedible','stale','nasty','bland',
        'rancid','junk','contaminated','lousy',"ugly","smelly","tasteless","sour","salty","bad",
        'worst','worse','sad','horrible','crazy']
    k = analyser.polarity_scores(original_review)
    rating=3
    if(k['neu']==1.0):
        z = int_features.split()
        for i in z:
            if i in positive_rest:
                rating=4
                break
            if i in negative_rest:
                rating=2
                break
    else:
        if((k['neg']>k['pos']) and ( (k['neg']-k['pos'])<=0.24 and (k['neg']-k['pos'])>=0  ) ):
            rating=3
        elif((k['neg']>k['pos']) and ( (k['neg']-k['pos'])>0.24 and (k['neg']-k['pos'])<=0.40  )):
            rating=2
        elif((k['neg']>k['pos']) and ( (k['neg']-k['pos'])>0.40)):
            rating=1
        elif((k['pos']>k['neg']) and ( (k['pos']-k['neg'])<=0.24 and (k['pos']-k['neg'])>=0  )):
            rating=3
        elif((k['pos']>k['neg']) and ( (k['pos']-k['neg'])>0.24 and (k['pos']-k['neg'])<=0.40  )):
            rating=4
        else:
            rating=5

    #score_numeric=analyser.polarity_scores(int_features)
    # food_count=0
    # hygiene_count=0
    # ambiance_count=0
    # pricing_count=0
    # miscelleanous_count=0
    # service_count=0

    mention_count={"food":[],"ambience":[],"service":[],"hygiene":[],"pricing":[],"miscelleanous":[]}
    temp_list=[]
    for j in review_tokenised:
        #temp_list=[]
        if j in wordsDictionary['food']:
            mention_count["food"].append(j)
            if("food" not in temp_list):
                temp_list.append("food")
                continue
        if j in wordsDictionary['service']:
            mention_count["service"].append(j)
            if("service" not in temp_list):
                temp_list.append("service")
                continue
        if j in wordsDictionary['hygiene']:
            mention_count["hygiene"].append(j)
            if("hygiene" not in temp_list):
                temp_list.append("hygiene")
                continue
        if j in wordsDictionary['pricing']:
            mention_count["pricing"].append(j)
            if("pricing" not in temp_list):
                temp_list.append("pricing")
                continue
        if j in wordsDictionary['ambiance']:
            mention_count["ambience"].append(j)
            if("ambience" not in temp_list):
                temp_list.append("ambience")
                continue
        if j in wordsDictionary['miscelleanous']:
            mention_count["miscelleanous"].append(j)
            if("miscelleanous" not in temp_list):
                temp_list.append("miscelleanous")
                continue
    if(len(temp_list)==0):
        temp_list.append("miscelleanous")
    #aspects=', '.join(temp_list)
    aspects=""
    for i in temp_list:
        aspects+=i
        if(len(mention_count[i])!=0):
            aspects+="("+', '.join(mention_count[i])
        aspects+=");\n\n  "
    aspects="Aspects mentioned are "+aspects

    return render_template('predict.html',predict_text='review is {}'.format(ans),score=rating,aspects=aspects,
        int_features="Entered Review : {}".format(original_review))

    # return render_template('predict.html',predict_text='review is {}'.format(ans),score=rating,aspects=aspects,
    # 	int_features="Entered Review : {}".format(original_review))


@app.route('/analysis')
def analysis():

    df = pd.read_csv('model-restaurant - restaurant.csv')

    positive_count=0
    negative_count=0
    for i in df["rating"]:
        if(i==0):
            negative_count+=1
        else:
            positive_count+=1

    """sentiment_count=np.array([positive_count,negative_count])
    sentiment_labels=["Positive","Negative"]
    pie_chart=plt.pie(sentiment_count,labels=sentiment_labels)
    #plt.show()
    #plt.savefig("C:/Users/KP Inani/Downloads/flask_app1/static/pie_chart.png")
    sentiment_count_list=[positive_count,negative_count]"""


    #df['remove_lower_punct'] = df['reviews'].str.lower().str.replace('[^\w\s]', ' ').str.replace(" \d+", " ").str.replace(' +', ' ').str.strip()
    df['remove_lower_punct'] = df['reviews'].str.lower().str.replace(',', ' ').str.replace("!", " ").str.replace('.', ' ').str.strip()
    #df['tokenise'] = df.apply(lambda row: nltk.word_tokenize(row[1]), axis=1)

    df['tokenise'] = df['remove_lower_punct'].apply(lambda row:row.split())	

    df['lemmatise'] = df['tokenise']
    
    ls=WordNetLemmatizer()

    for i in range(len(df['lemmatise'])):
        review=df['lemmatise'][i]
        for j in review:
            temp_review=[]
            if(j in contractions.keys()):
                temp_review.append(contractions[j])
            else:
                temp_review.append(j)
        review=' '.join(temp_review)
        review=review.split()
        review=[word for word in review if word not in replace_list]
        review=[ls.lemmatize(word) for word in review]
        #df["lemmatise"][i]=review
        

    """fw=pd.read_csv("final words.csv")
    #print(fw.head())
    food=fw["food"].tolist()
    for i in range(len(food)):
        food[i]=food[i][1:-1].lower()
    #print(food[:5],len(food))
    x=fw["pricing"].tolist()
    pricing=[]
    for i in range(len(x)):
        if(str(x[i])=="nan"):
            break
        pricing.append(x[i][1:-1].lower())
    #print(pricing[:5],len(pricing))
    x=fw["hygiene"].tolist()
    hygiene=[]
    for i in range(len(x)):
        if(str(x[i])=="nan"):
            break
        hygiene.append(x[i][1:-1].lower())
    #print(hygiene[:5],len(hygiene))
    x=fw["service"].tolist()
    service=[]
    for i in range(len(x)):
        if(str(x[i])=="nan"):
            break
        service.append(x[i][1:-1].lower())
    #print(service[:5],len(service))
    x=fw["ambiance"].tolist()
    ambiance=[]
    for i in range(len(x)):
        if(str(x[i])=="nan"):
            break
        ambiance.append(x[i][1:-1].lower())
    #print(ambiance[:5],len(ambiance))
    x=fw["miscelleanous"].tolist()
    miscelleanous=[]
    for i in range(len(x)):
        if(str(x[i])=="nan"):
            break
        miscelleanous.append(x[i][1:-1].lower())
    #print(miscelleanous[:5],len(miscelleanous))
    food=list(set(food))
    #print(food[:5],len(food))
    miscelleanous=list(set(miscelleanous))
    #print(miscelleanous[:5],len(miscelleanous))
    hygiene=list(set(hygiene))
    #print(hygiene[:5],len(hygiene))
    service=list(set(service))
    #print(service[:5],len(service))
    pricing=list(set(pricing))
    #print(pricing[:5],len(pricing))
    ambiance=list(set(ambiance))
    #print(ambiance[:5],len(ambiance))"""

    food_count=0
    hygiene_count=0
    ambiance_count=0
    pricing_count=0
    miscelleanous_count=0
    service_count=0
    # food_score=0
    # service_score=0
    # hygiene_score=0
    # ambiance_score=0
    # pricing_score=0
    # miscelleanous_score=0
    mention_count={"food":{},"ambiance":{},"service":{},"hygiene":{},"pricing":{},"miscelleanous":{}}
    df['aspects']=df.apply(lambda _: '', axis=1)
    for i in range(len(df["lemmatise"])):
        temp_list=[]
        for j in df["lemmatise"][i]:
            if j in wordsDictionary['food']:
                mention_count["food"][j]=mention_count["food"].get(j,0)+1
                if("food" not in temp_list):
                    temp_list.append("food")
                    food_count+=1
                    #food_score+=df["sentiment score"][i]
            if j in wordsDictionary['hygiene']:
                mention_count["hygiene"][j]=mention_count["hygiene"].get(j,0)+1
                if("hygiene" not in temp_list):
                    temp_list.append("hygiene")
                    hygiene_count+=1
                    #hygiene_score+=df["sentiment score"][i]
            if j in wordsDictionary['service']:
                mention_count["service"][j]=mention_count["service"].get(j,0)+1
                if("service" not in temp_list):
                    temp_list.append("service")
                    service_count+=1
                    #service_score+=df["sentiment score"][i]
            if j in wordsDictionary['pricing']:
                mention_count["pricing"][j]=mention_count["pricing"].get(j,0)+1
                if("pricing" not in temp_list):
                    temp_list.append("pricing")
                    pricing_count+=1
                    #pricing_score+=df["sentiment score"][i]
            if j in wordsDictionary['ambiance']:
                mention_count["ambiance"][j]=mention_count["ambiance"].get(j,0)+1
                if("ambiance" not in temp_list):
                    temp_list.append("ambiance")
                    ambiance_count+=1
                    #ambiance_score+=df["sentiment score"][i]
            if j in wordsDictionary['miscelleanous']:
                mention_count["miscelleanous"][j]=mention_count["miscelleanous"].get(j,0)+1
                if("miscelleanous" not in temp_list):
                    temp_list.append("miscelleanous")
                    miscelleanous_count+=1
                    #miscelleanous_score+=df["sentiment score"][i]
        if(len(temp_list)==0):
            temp_list.append("miscelleanous")
            miscelleanous_count+=1
            #miscelleanous_score+=df["sentiment score"][i] 
        #df["aspects"][i] = temp_list


    aspect_labels=["Food","Service","Ambience","Pricing","Hygiene","Miscelleanous"]
    aspect_counts=[food_count,service_count,ambiance_count,pricing_count,hygiene_count,miscelleanous_count]
    # plt.bar(aspect_labels,aspect_counts)
    # plt.title("Count of Aspects Mentioned in reviews")
    # plt.xlabel("Aspects")
    # plt.ylabel("Count of Reviews in which said aspect is mentioned")
    #plt.show()

    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'pie_chart.jpeg')

    # with open('C:/Users/KP Inani/Downloads/flask_app1/template/mention_count.json', 'w') as outfile:
    # 	json.dump(mention_count, outfile)
    wordcloud_food=""
    for k,v in mention_count["food"].items():
        for j in range(v):
            wordcloud_food+=k+" "
    wordcloud_ambiance=""
    for k,v in mention_count["ambiance"].items():
        for j in range(v):
            wordcloud_ambiance+=k+" "
    wordcloud_service=""
    for k,v in mention_count["service"].items():
        for j in range(v):
            wordcloud_service+=k+" "
    wordcloud_pricing=""
    for k,v in mention_count["pricing"].items():
        for j in range(v):
            wordcloud_pricing+=k+" "
    wordcloud_hygiene=""
    for k,v in mention_count["hygiene"].items():
        for j in range(v):
            wordcloud_hygiene+=k+" "
    wordcloud_miscelleanous=""
    for k,v in mention_count["miscelleanous"].items():
        for j in range(v):
            wordcloud_miscelleanous+=k+" "

    # with open('C:/Users/KP Inani/Downloads/flask_app1/template/mention_count.json', 'w') as outfile:
    #  	json.dump(wordcloud_food, outfile)

    return render_template('analysis.html',pos_neg_count=[positive_count,negative_count],aspect_counts=aspect_counts,
        mention_count=mention_count,wordcloud_food=[wordcloud_food],
        wordcloud_ambiance=[wordcloud_ambiance],wordcloud_pricing=[wordcloud_pricing],
        wordcloud_service=[wordcloud_service],wordcloud_hygiene=[wordcloud_hygiene],
        wordcloud_miscelleanous=[wordcloud_miscelleanous])

if __name__ == '__main__':
    app.run(debug=True)
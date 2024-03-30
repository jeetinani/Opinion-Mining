import numpy as np
import pandas as pd
from flask import Flask,render_template,request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from helpers.Words import getMentionsandCount, getTotal
from helpers.Transformer import getTransformedReview, getTokenisedReview
from models.Model import miner, tfidfVectorizer
#import json
#PEOPLE_FOLDER = os.path.join('static', 'people_photo')

# nltk.download('stopwords')
#stopwords = stopwords.words('english')


app=Flask(__name__,template_folder='template',static_folder='static')

#wordsDictionary = getWordsDictionary()
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



    #dataSet = pd.read_csv('model-restaurant - restaurant.csv')
    #df = df.rename(columns={'text':'reviews'})


    #df['reviews'].isna().sum()
    #df['remove_lower_punct'] = df['reviews']

    analyser = SentimentIntensityAnalyzer()

    
    #corpus=[]

    #ls=WordNetLemmatizer()
    
    """ for review in dataSet['reviews']:
        corpus.append(getTransformedReview(review))
     """

    #df.to_csv("trial.csv")
    
    """ tfidfVectorizer=TfidfVectorizer(max_features=15000)
    X=tfidfVectorizer.fit_transform(corpus).toarray()
    y=dataSet.iloc[:,1].values """

  
    #from sklearn.model_selection import train_test_split
    #from sklearn.metrics import confusion_matrix,accuracy_score
    #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    #from sklearn.naive_bayes import MultinomialNB
    #miner=MultinomialNB().fit(X_train,y_train)

    original_review=request.form['reviews']
    transformedReview = getTransformedReview(original_review)
    
    #miner=MultinomialNB().fit(X,y)

    tester = np.array([transformedReview])
    vector = tfidfVectorizer.transform(tester)
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
        z = transformedReview.split()
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

    #aspects=', '.join(temp_list)
    mentions = getMentionsandCount([getTokenisedReview(original_review)])
    aspectResponse=[]
    for key,value in mentions.items() :
        if(len(value)>0):
            aspectResponse.append(key+'('+','.join(value.keys())+')')
    if(len(aspectResponse)==0):
        aspectResponse.append('Miscelleanous')
            

    return render_template('predict.html',predict_text='Review is {}'.format(ans),score=rating,aspects="Aspects mentioned are : {}".format(','.join(aspectResponse)),
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

    food_count=0
    hygiene_count=0
    ambiance_count=0
    pricing_count=0
    miscelleanous_count=0
    service_count=0

    """ mention_count={"food":{},"ambiance":{},"service":{},"hygiene":{},"pricing":{},"miscelleanous":{}}
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
        #df["aspects"][i] = temp_list """

    corpus=[]
    for review in df['reviews']:
        corpus.append(getTokenisedReview(review))
        
    mention_count = getMentionsandCount(corpus)

    aspect_labels=["Food","Service","Ambience","Pricing","Hygiene","Miscelleanous"]
    aspect_counts=[getTotal(mention_count['food']),
                   getTotal(mention_count['service']),
                   getTotal(mention_count['ambiance']),
                   getTotal(mention_count['pricing']),
                   getTotal(mention_count['hygiene']),
                   getTotal(mention_count['miscelleanous'])]
    # plt.bar(aspect_labels,aspect_counts)
    # plt.title("Count of Aspects Mentioned in reviews")
    # plt.xlabel("Aspects")
    # plt.ylabel("Count of Reviews in which said aspect is mentioned")
    #plt.show()

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
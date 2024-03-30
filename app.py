import numpy as np
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from flask import Flask,render_template,request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from helpers.Words import replace_list, contractions, wordsDictionary, getMentions
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

    """ mentions={"food":[],"ambience":[],"service":[],"hygiene":[],"pricing":[],"miscelleanous":[]}
    temp_list=[]
    for j in getTokenisedReview(original_review):
        #temp_list=[]
        if j in wordsDictionary['food']:
            mentions["food"].append(j)
            if("food" not in temp_list):
                temp_list.append("food")
                continue
        if j in wordsDictionary['service']:
            mentions["service"].append(j)
            if("service" not in temp_list):
                temp_list.append("service")
                continue
        if j in wordsDictionary['hygiene']:
            mentions["hygiene"].append(j)
            if("hygiene" not in temp_list):
                temp_list.append("hygiene")
                continue
        if j in wordsDictionary['pricing']:
            mentions["pricing"].append(j)
            if("pricing" not in temp_list):
                temp_list.append("pricing")
                continue
        if j in wordsDictionary['ambiance']:
            mentions["ambience"].append(j)
            if("ambience" not in temp_list):
                temp_list.append("ambience")
                continue
        if j in wordsDictionary['miscelleanous']:
            mentions["miscelleanous"].append(j)
            if("miscelleanous" not in temp_list):
                temp_list.append("miscelleanous")
                continue
    if(len(temp_list)==0):
        temp_list.append("miscelleanous") """
    #aspects=', '.join(temp_list)
    """ aspects=""
    for i in temp_list:
        aspects+=i
        if(len(mentions[i])!=0):
            aspects+="("+', '.join(mentions[i])
        aspects+=");\n\n  " """
    mentions = getMentions(getTokenisedReview(original_review))
    aspectResponse=[]
    for key,value in mentions.items() :
        if(len(value)>0):
            aspectResponse.append(key+'('+','.join(value)+')')
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

    """sentiment_count=np.array([positive_count,negative_count])
    sentiment_labels=["Positive","Negative"]
    pie_chart=plt.pie(sentiment_count,labels=sentiment_labels)
    #plt.show()
    #plt.savefig("pie_chart.png")
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
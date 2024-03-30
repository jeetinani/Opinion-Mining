import numpy as np
import pandas as pd
from flask import Flask,render_template,request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from helpers.Words import getMentionsandCount, getTotal, positive_rest, negative_rest
from helpers.Transformer import getTransformedReview, getTokenisedReview
from models.Model import miner, tfidfVectorizer

app=Flask(__name__,template_folder='template',static_folder='static')

@app.route('/')
def  home_default():
    return render_template('home.html')

@app.route('/home')
def  home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    analyser = SentimentIntensityAnalyzer()

    original_review=request.form['reviews']
    transformedReview = getTransformedReview(original_review)

    tester = np.array([transformedReview])
    vector = tfidfVectorizer.transform(tester)
    prediction = miner.predict(vector)
    if(prediction == 0):
        ans="Negative"
    else:
        ans="Positive"

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

    mentions = getMentionsandCount([getTokenisedReview(original_review)])
    aspectResponse=[]
    for key,value in mentions.items() :
        if(len(value)>0):
            aspectResponse.append(key+'('+','.join(value.keys())+')')
    if(len(aspectResponse)==0):
        aspectResponse.append('Miscelleanous')
            

    return render_template('predict.html',predict_text='Review is {}'.format(ans),score=rating,aspects="Aspects mentioned are : {}".format(','.join(aspectResponse)),
        int_features="Entered Review : {}".format(original_review))


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

    corpus=[]
    for review in df['reviews']:
        corpus.append(getTokenisedReview(review))
        
    mention_count = getMentionsandCount(corpus)

    aspect_counts=[getTotal(mention_count['food']),
                   getTotal(mention_count['service']),
                   getTotal(mention_count['ambiance']),
                   getTotal(mention_count['pricing']),
                   getTotal(mention_count['hygiene']),
                   getTotal(mention_count['miscelleanous'])]
    #aspect_labels=["Food","Service","Ambience","Pricing","Hygiene","Miscelleanous"]
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

    return render_template('analysis.html',pos_neg_count=[positive_count,negative_count],aspect_counts=aspect_counts,
        mention_count=mention_count,wordcloud_food=[wordcloud_food],
        wordcloud_ambiance=[wordcloud_ambiance],wordcloud_pricing=[wordcloud_pricing],
        wordcloud_service=[wordcloud_service],wordcloud_hygiene=[wordcloud_hygiene],
        wordcloud_miscelleanous=[wordcloud_miscelleanous])

if __name__ == '__main__':
    app.run(debug=True)
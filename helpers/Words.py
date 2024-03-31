import pandas as pd

finalWords=pd.read_csv('datasheets/shared words list.csv')
#print(finalWords.head())

positive_rest = ('good','amazing','awesome','great','tasty','divine','delicious','yumm','yummy','fingerlicking',
    'heavenly','appetizing','flavorful','palatable','flavorful','flavorsome','good-tasting','superb',
    'fingerlicking','flawless','beautiful',"garnishing","garnished","sweet","savoury",'enjoy','enjoyed')
negative_rest = ('old','yuck','ewww','expensive','costly','inedible','stale','nasty','bland',
    'rancid','junk','contaminated','lousy',"ugly","smelly","tasteless","sour","salty","bad",
    'worst','worse','sad','horrible','crazy')

replace_list=('i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
            'can', 'will', 'just', 'now')

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

def getTotal(dic):
    sum = 0
    for count in dic.values():
        sum = sum + count
    return sum

def getList(category):
    words=finalWords[category].tolist()
    returnWords = []
    for i in range(len(words)):
        if(str(words[i])=="nan"):
            break
        returnWords.append(words[i][1:-1].lower())
    return list(set(returnWords))

def getWordsDictionary():
    
    wordsDictionary = {}
    #print(finalWords.head())
    categories = ['food','service','ambiance','pricing','hygiene','miscelleanous']
    for category in categories :
        wordsDictionary[category] = getList(category)
    #print(wordsDictionary)
    return wordsDictionary
    #print(fw.head())
    food=finalWords["food"].tolist()
    for i in range(len(food)):
        food[i]=food[i][1:-1].lower()
        #print(food[:5],len(food))
    x=finalWords["pricing"].tolist()
    pricing=[]
    for i in range(len(x)):
        if(str(x[i])=="nan"):
            break
        pricing.append(x[i][1:-1].lower())
        #print(pricing[:5],len(pricing))
    x=finalWords["hygiene"].tolist()
    hygiene=[]
    for i in range(len(x)):
        if(str(x[i])=="nan"):
            break
        hygiene.append(x[i][1:-1].lower())
        #print(hygiene[:5],len(hygiene))
    x=finalWords["service"].tolist()
    service=[]
    for i in range(len(x)):
        if(str(x[i])=="nan"):
            break
        service.append(x[i][1:-1].lower())
        #print(service[:5],len(service))
    x=finalWords["ambiance"].tolist()
    ambiance=[]
    for i in range(len(x)):
        if(str(x[i])=="nan"):
            break
        ambiance.append(x[i][1:-1].lower())
        #print(ambiance[:5],len(ambiance))
    x=finalWords["miscelleanous"].tolist()
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
        #print(ambiance[:5],len(ambiance))

wordsDictionary = getWordsDictionary()

def getMentionsandCount(reviews):
    
    mentions={"food":{},"ambiance":{},"service":{},"hygiene":{},"pricing":{},"miscelleanous":{}}
    for review in reviews:
        for j in review:
            if j in wordsDictionary['food']:
                mentions["food"][j]=mentions["food"].get(j,0)+1    
            if j in wordsDictionary['service']:
                mentions["service"][j]=mentions["service"].get(j,0)+1
            if j in wordsDictionary['hygiene']:
                mentions["hygiene"][j]=mentions["hygiene"].get(j,0)+1
            if j in wordsDictionary['pricing']:
                mentions["pricing"][j]=mentions["pricing"].get(j,0)+1
            if j in wordsDictionary['ambiance']:
                mentions["ambiance"][j]=mentions["ambiance"].get(j,0)+1
            if j in wordsDictionary['miscelleanous']:
                mentions["miscelleanous"][j]=mentions["miscelleanous"].get(j,0)+1
    return mentions
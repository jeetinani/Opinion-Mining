import re
from helpers.Words import replace_list, contractions
from nltk.stem import WordNetLemmatizer
def getTokenisedReview(review):
    return re.sub('[^a-zA-z]',' ',review).strip().split()

def getTransformedReview(review):

    review = getTokenisedReview(review)
    wordNetLemmatizer=WordNetLemmatizer()
    temp_review=[]
    for word in review:
        if(word in contractions.keys()):
            temp_review.append(contractions[word])
        else:
            temp_review.append(word)
    temp_review=[wordNetLemmatizer.lemmatize(word) for word in temp_review if word not in replace_list]
    review=' '.join(temp_review)
    return review
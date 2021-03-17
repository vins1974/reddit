from flask import Flask, render_template
import flask
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np
import praw
from textblob import TextBlob
import sys
import xgboost

app = Flask(__name__)
def Subjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def Polarity(text):
    return TextBlob(text).sentiment.polarity
def word_count(text):
    wordList = re.sub("[^\w]", " ", text).split()
    return len(wordList)
def clean_message(text):
    text = re.sub(r'[^\w\s]', '', text)
    l_text = " ".join(word for word in text.lower().split() if word not in ENGLISH_STOP_WORDS)
    return l_text
     
# for sentimental analsysis
with open('senti.pkl', "rb") as f:
    senti = pickle.load(f)
    
# One hot encoding
with open('one_hot.pkl', "rb") as f:
    enc = pickle.load(f)
    
# for model Prediction
xgb_b=xgboost.Booster()
xgb_b.load_model('xgbr.booster')


# To get information for the Reddit url
def extract_data(url):
    data = {}
    reddit = praw.Reddit(client_id='WUTH6H3Cx7KW4w',
                         client_secret='hBOWXZ37WOOY9M9oT-SD-2H7ql_7HQ',
                         user_agent='user_agent')
    sub_data = reddit.submission(url=str(url))
    data['body'] = [str(sub_data.title)]
    data['downs'] = sub_data.downs
    data['upvote_ratio']=sub_data.upvote_ratio
    data['gilded'] = [sub_data.gilded]
    data['word_count'] = word_count(sub_data.title)
    data['over_18'] = [sub_data.over_18]
    data['number_of_Comments'] = [sub_data.num_comments]
    data['Subjectivity'] = Subjectivity(sub_data.title)
    data['Polarity'] = Polarity(sub_data.title)
    scores = senti.polarity_scores(sub_data.title)
    data['Compound'] = scores['compound']
    data['neg'] = scores['neg']
    data['neu'] = scores['neu']
    data['pos'] = scores['pos']
    df = pd.DataFrame(data)
        
    return df

@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = str(flask.request.form['url'])
    data = extract_data(url)
    Body = clean_message(data['body'][0])
    # Converting word2vector
    df_word_token = pd.read_csv('word_token.csv')
    print(' df_word_token loaded')
    sys.stdout.flush()
    test_title = []
    for word in Body.split():
        if word in df_word_token.columns:
            test_title.append(df_word_token[word])
    max_len = 300
    test_title = test_title + [0] * (max_len - len(test_title))
    embed_mat = np.array(pd.read_csv('embed_mat.csv', sep=' '))
    vectors = []
    for n in test_title:
        vectors.append(embed_mat[n])
    vectors = [item for sublist in vectors for item in sublist]
    arr = np.array(vectors)
    final_vector = np.mean(arr, axis=0)
    df_test_body = pd.DataFrame(np.array(final_vector)).T
    
    # one hot encoding with column names
    categories = ['over_18']
    test_encoded = enc.transform(data[categories])  
    col_names = [False, True]
    test_ohe = pd.DataFrame(test_encoded.todense(), columns=col_names)
    
    data.drop(["body", 'over_18'], axis=1, inplace=True)
    data.reset_index(inplace=True, drop=True)
    X_test = pd.concat([data, df_test_body, test_ohe], axis=1)

    #  Predict with XGBoosting Regressor
    score = xgb_b.predict(xgboost.DMatrix(X_test))
    return render_template('Index.html', score='Predicted score for the given Reddit post is: {}'.format(score ))


if __name__ == "__main__":
    app.run(debug=True)
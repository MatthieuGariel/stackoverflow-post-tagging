from flask import Flask
app = Flask(__name__)
app.debug = True

from flask import render_template, request
from flask import jsonify
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
from nltk.corpus import stopwords
import csv
import joblib
from nltk import word_tokenize


# model = joblib.load("model_w_best_params.sav")
# scaler_imported = joblib.load("scaler.sav")

stop = stopwords.words('english')
stemmer = LancasterStemmer()
vectorizer = joblib.load("tfidf_vectorizer")


@app.route('/', methods=['GET'])
def index(): return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        result = request.form
        title = result['post_title']
        body = result['post_text']
        titleAndBody = str(title) + " " + str(body)
        corpus = word_tokenize(titleAndBody)
        corpus = [word.lower() for word in corpus]
        corpus_wo_stopwords = list(filter(lambda x: x not in stop, corpus))
        corpus_wo_stopwords = [i for i in corpus_wo_stopwords if i.isalpha()]
        corpus_wo_stopwords = [word.lower() for word in corpus_wo_stopwords]
        corpus_lanc_stemmed = [stemmer.stem(word) for word in corpus_wo_stopwords]
        corpus_lanc_stemmed = ' '.join(corpus_lanc_stemmed)
        corpus_lanc_stemmed_transformed = vectorizer.transform(corpus_lanc_stemmed)
        return (str(corpus_lanc_stemmed_transformed))
    # else:
    #    return (request.form)

if __name__ == '__main__':
    app.run()

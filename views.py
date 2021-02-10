from flask import Flask
app = Flask(__name__)
app.debug = True

from flask import render_template, request
from flask import jsonify
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import csv
import joblib
from nltk import word_tokenize

stop = stopwords.words('english')
stemmer = LancasterStemmer()
vectorizer = joblib.load("tfidf_vectorizer")
BR_logreg = joblib.load("BR_logreg")
tags = np.genfromtxt('tags.csv', dtype=str, delimiter=",")


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
        corpus_lanc_stemmed = [stemmer.stem(word) for word in corpus_wo_stopwords]
        corpus_lanc_stemmed = ';'.join(corpus_lanc_stemmed)
        corpus_lanc_stemmed = [corpus_lanc_stemmed]
        corpus_lanc_stemmed_transformed = vectorizer.transform(corpus_lanc_stemmed)
        # return(str((np.round(BR_logreg.predict_proba(corpus_lanc_stemmed_transformed.toarray()).toarray(),
        # 2))[0]))
        return(str(tags[(np.round(BR_logreg.predict_proba(corpus_lanc_stemmed_transformed.toarray()).toarray(), 2) > 0.01)[0]]))

if __name__ == '__main__':
    app.run()

from flask import Flask
app = Flask(__name__)

from flask import render_template, request
from flask import jsonify
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
from nltk.corpus import stopwords
import csv
import joblib


# df = pd.read_csv('data.csv', delimiter=',')
# df.drop(df.columns[[0]], axis=1, inplace=True)

# model = joblib.load("model_w_best_params.sav")
# scaler_imported = joblib.load("scaler.sav")

stop = stopwords.words('english')
stemmer = LancasterStemmer()
# vectorizer_imported = joblib.load("scaler.sav")


@app.route('/', methods=['GET'])
def index(): return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        titleAndBody = result["post_title"] + " " + result["post_text"]
        corpus = nltk.word_tokenize(titleAndBody)
        corpus = corpus.lower()
        corpus_wo_stopwords = list(filter(lambda x: x not in stop, corpus))
        corpus_wo_stopwords = [i for i in corpus_wo_stopwords if not x.isalpha()]
        corpus_wo_stopwords = corpus_wo_stopwords.lower()
        corpus_lanc_stemmed = [stemmer.stem(word) for word in corpus_wo_stopwords]
        corpus_lanc_stemmed = ' '.join(corpus_lanc_stemmed)
        # corpus_lanc_stemmed_transformed = scaler_imported.transform(corpus_lanc_stemmed)
        return (str(corpus_lanc_stemmed))

if __name__ == '__main__':
    app.run()

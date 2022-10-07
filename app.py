from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# ML Pkg
# from sklearn.externals import joblib
import joblib
import time
from PIL import Image

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("data/names_dataset.csv")
    #etiquetas y caracteristicas
    df_x = df.name
    df_y = df.sex

    corpus = df_x
    cv = CountVectorizer()
    x = cv.fit_transform(corpus)

    naivebayes_model = open("data/naivebayesgendermodel.pkl","rb")
    clf = joblib.load(naivebayes_model)

    if request.method == 'POST':
        namequery = request.form['namequery']
        data = [namequery]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('results.html', prediction = my_prediction, name = namequery.upper())

if __name__ == '__main__':
    app.run(debug=True)


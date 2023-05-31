import csv
import numpy as np
from flask import Flask, request, render_template
from predict import recommendation, get_data

app = Flask(__name__)

@app.route('/')
def home():
    header, _, _ = get_data()
    return render_template('index.html', features=header)

@app.route('/result',methods=['POST'])
def predict():
    features = [v for v in request.form.values()]
    res = recommendation(features)
    return render_template('result.html', data=res, features=features)

if __name__ == "__main__":
    app.run()
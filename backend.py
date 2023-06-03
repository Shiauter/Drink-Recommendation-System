import csv
import numpy as np
from flask import Flask, request, render_template
from predict import recommendation, get_data
import os

os.environ["OMP_NUM_THREADS"] = '1'

app = Flask(__name__)

@app.route('/')
def home():
    header, _, _, _ = get_data()
    return render_template('index.html', features=header[:-3])

@app.route('/result',methods=['POST'])
def predict():
    features = [v for v in request.form.values()]
    res = recommendation(features)
    return render_template('result.html', data=res, features=features)

if __name__ == "__main__":
    app.run(debug=True)
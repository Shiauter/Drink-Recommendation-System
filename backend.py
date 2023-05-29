import csv
import numpy as np
from flask import Flask, request, render_template
import pickle
from predict import recommendation

app = Flask(__name__)

@app.route('/')
def home():
    # read feature data here
    with open("item_attributes.csv", "r") as file:
        rows = csv.reader(file, delimiter=',')
        header = next(rows)
    return render_template('index.html', features=header[1:])

@app.route('/result',methods=['POST'])
def predict():
    features = [v for v in request.form.values()]
    print(features)
    res = recommendation(features)
    return render_template('result.html', data=res, features=features)

if __name__ == "__main__":
    app.run(debug=True)
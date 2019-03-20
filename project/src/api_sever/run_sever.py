import sys
sys.path.append('..')

from flask import Flask, request
import pandas as pd
from model.predict import estimator

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def execPredict():
        title='classify'
        if request.method == 'POST':
                testData = request.json['data']
                model = estimator()
                testData = model.transformData(testData)
                score = model.execPredict(testData)

        return score

if __name__ == '__main__':
        app.run(debug=True, host='localhost', port=5000, threaded=True)
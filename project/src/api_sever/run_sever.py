from flask import Flask, request
import pandas as pd

app = Flask(__name__)

@app.route('/', methods['POST'])
def execPredict():
    title='classify'
    if request.method = 'POST':
        jsonData = request.json()
        testData = parseJson2Dataframe(jsonData)

def parseJson2Dataframe(jsonData):
    data = pd.read_json(jsonData)
    return data

if __name__='__main__':
    app.run(debug=True, host='0.0.0.0', port=8888, threaded=True)
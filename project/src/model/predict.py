from sklearn import linear_model
from sklearn.model_selection import KFold, cross_validate
import pandas as pd
import numpy as np
import pickle

MEMBER_NAME = [
    "hanako",
    "hanao",
    "fritarou",
]

class predict:
    def loadModel(self):
        fileName = "model.pkl"
        clf = pickle.load(open(fileName, 'rb'))
        print(clf)
        return clf

    def makeTestData(self,rowData):
        makeTestData = pd.read_json(rowData)
        return makeTestData

    def execPredict(self, clf, testData):
        score = clf.predict(testData)
        result = pd.dataFrame(
                    data=score,
                    columns=MEMBER_NAME)

        resJ = result.to_json()

        return resJ




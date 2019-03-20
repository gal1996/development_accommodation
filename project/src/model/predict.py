from sklearn import linear_model
from sklearn.model_selection import KFold, cross_validate
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('..')
from model.dataAugmentation import DataAugmentation

def makeTestData():
    data  = [
        [0.7, 4, 15, 5, 9, 10],
        [0.8, 4, 14, 4, 9, 9],
        [0.8, 4, 12, 4, 9, 9],
        [0.6, 3, 13, 5, 10, 9],
        [0.7, 4, 10, 4, 10, 10],
        [0.7, 2, 8, 5, 10, 9],
        [0.8, 3, 15, 3, 5, 7],
        [0.5, 3, 7, 5, 5, 5],
        [1.0, 5, 15, 5, 10, 10],
        [-0.3, 1, 1, 1, 1, 1],
    ]

    data = pd.DataFrame(data=data, columns=['laughStd', 'rareEncountPoint', 'takenPictureWithManyPeaplePoint', 'takeGoodPicturePoint',
                    'betweenProductInteractPoint', 'diversityPoint'])
    jsonData = data.to_json()

    return data


class estimator:
    def __init__(self):
        self.clf = self.loadModel()

    def loadModel(self):
        fileName = "../../model_file/model.pkl"
        clf = pickle.load(open(fileName, 'rb'))
        return clf

    def execPredict(self, data):
        print(data)
        d = DataAugmentation()
        testData = d.makeNewFeature(data)
        print(testData)
        testData = d.execStd(testData)
        print(testData)
        testData = d.execPca(testData)

        score = self.clf.predict(testData)
        print(score)
        result = pd.DataFrame(data=score,columns=['data'])

        resJ = result.to_json()

        return resJ

    def transformData(self, data):
        data = pd.DataFrame(data=data, columns=['laughStd', 'rareEncountPoint', 'takenPictureWithManyPeaplePoint', 'takeGoodPicturePoint',
                'betweenProductInteractPoint', 'diversityPoint'])

        return data


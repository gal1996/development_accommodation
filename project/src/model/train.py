from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, SGDRegressor, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import KFold, cross_validate

import pandas as pd
import numpy as np
import pickle
import json
from dataAugmentation import DataAugmentation

class train:
    def makeTrainData(self):
        #得られたデータ
        data = [
            [1.0, 5, 15, 5, 10, 10],
            [0.9, 4, 14, 5, 10, 10],
            [0.9, 5, 12, 5, 9, 10],
            [0.9, 4, 13, 5, 10, 10],
            [0.8, 4, 14, 5, 10, 10],
            [0.7, 4, 13, 5, 9, 10],
            [0.9, 5, 13, 5, 10, 10],
            [0.8, 4, 15, 4, 10, 9],
            [0.8, 4, 12, 5, 9, 10],
            [0.7, 3, 13, 5, 10, 9],
            [0.7, 4, 15, 5, 9, 10],
            [0.8, 4, 14, 4, 9, 9],
            [0.6, 3, 12, 4, 7, 6],
            [0.8, 4, 12, 4, 9, 9],
            [0.7, 4, 13, 4, 9, 7],
            [0.6, 3, 13, 5, 10, 9],
            [0.4, 3, 11, 4, 6, 7],
            [0.7, 4, 10, 4, 10, 10],
            [0.5, 3, 8, 3, 7, 4],
            [0.7, 2, 8, 5, 10, 9],
            [0.8, 3, 15, 3, 5, 7],
            [0.5, 3, 7, 3, 5, 5],
            [0.8, 0, 5, 5, 5, 4],
            [0.1, 2, 4, 3, 10, 10],
            [0.5, 5, 12, 5, 2, 2],
            [0.9, 2, 11, 1, 8, 6],
            [0.5, 5, 5, 0, 5, 5],
            [0.3, 2, 4, 1, 3, 4],
            [-0.4, 0, 2, 0, 2, 3],
            [-0.1, 1, 3, 2, 5, 6],
            [0.0, 2, 3, 3, 2, 1],
            [0.1, 1, 4, 3, 1, 1],
            [0.3, 2, 3, 0, 2, 5],
            [0.6, 5, 7, 4, 6, 6],
            [0.1, 2, 4, 1, 4, 5],
            [0.3, 3, 5, 2, 5, 5],
            [0.4, 1, 4, 1, 6, 2],
            [0.5, 3, 7 ,3, 4, 9],
            [0.3, 4, 6, 1, 2, 2],
            [0.8, 5, 10, 3, 8, 8],
            [0.1, 2, 3, 3, 4, 4],
            [0.8, 4, 12, 3, 8, 8],
            [0.4, 2, 6, 2, 3, 2],
            [0.7, 4, 10, 3, 7, 6],
            [-0.3, 2, 4, 1, 3, 2],
            [0.9, 5, 12, 5, 9, 10],
            [0.0, 2, 3, 0, 3, 2],
            [0.7, 5, 15, 5, 10, 10],
            [-0.2, 1, 4, 0, 3, 2],
            [0.7, 3, 9, 4, 7, 6],
        ]

        #データ形成
        y_data = pd.DataFrame(
            data=data,
            columns=['laughStd', 'rareEncountPoint', 'takenPictureWithManyPeaplePoint', 'takeGoodPicturePoint',
                    'betweenProductInteractPoint', 'diversityPoint'])
        #投票により得られた得点
        y_label = [10, 9, 8, 8, 8, 7, 8, 6, 6, 6, 7, 7, 6, 10, 7, 6, 6, 8, 5, 5, 2, 5, 3, 0, 2, 0, 1, 0, 2, 1, 4, 5, 10, 4, 5, 4, 8, 2, 8, 1, 2, 10, 4, 9, 2, 11, 3, 9, 4, 7]
        y_dict = {}

        return y_data, y_label


    def makeClassifier(self, trainData, trainLabel):
        d = DataAugmentation()
        data = d.makeNewFeature(trainData)
        data = d.execStd(data)

        clf1 = KNeighborsRegressor(n_neighbors=2)
        clf2 = LogisticRegression()
        clf3 = SVR()
        sclf = StackingRegressor(regressors=[clf1, clf2, clf3], meta_regressor=clf1)
        fittedClf = self.execTrain(sclf, data, trainLabel)

        self.saveModel(fittedClf)

        self.execCrossValidate(fittedClf, data, trainLabel)

    def execTrain(self, clf, trainData, trainLabel):
        clf.fit(trainData, trainLabel)
        return clf

    def saveModel(self, clf):
        fileName = '../../model_file/model.pkl'
        pickle.dump(clf, open(fileName, 'wb'))

    def execCrossValidate(self, clf, trainData, trainLabel):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(clf, trainData, trainLabel, cv=kf, return_train_score=True)
        print(scores['train_score'].mean())

def main():
    model = train()
    trainData, trainLabel = model.makeTrainData()

    clf = model.makeClassifier(trainData, trainLabel)

if  __name__ == '__main__':
    main()


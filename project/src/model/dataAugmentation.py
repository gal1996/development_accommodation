import pandas as pd
import numpy as np

def makeTrainData():
        #得られたデータ
        data = [
            [1.0, 5, 15, 10, 10, 10],
            [0.9, 5, 12, 10, 9, 10],
            [0.8, 4, 14, 10, 10, 10],
            [0.9, 5, 13, 9, 10, 10],
            [0.8, 4, 12, 9, 9, 10],
            [0.7, 4, 15, 9, 9, 10],
            [0.8, 4, 14, 8, 9, 9],
            [0.8, 4, 12, 8, 9, 9],
            [0.6, 3, 13, 9, 10, 9],
            [0.7, 4, 10, 7, 10, 10],
            [0.7, 2, 8, 10, 10, 9],
            [0.8, 3, 15, 5, 5, 7],
            [0.5, 3, 7, 5, 5, 5],
            [0.8, 0, 5, 10, 5, 4],
            [0.1, 2, 4, 5, 10, 10],
            [0.5, 5, 12, 10, 2, 2],
            [0.9, 2, 11, 2, 8, 6],
            [0.5, 5, 5, 0, 5, 5],
            [0.3, 2, 4, 2, 3, 4],
            [-0.4, 0, 2, 1, 2, 3],
        ]

        #データ形成
        y_data = pd.DataFrame(
            data=data,
            columns=['laughStd', 'rareEncountPoint', 'takenPictureWithManyPeaplePoint', 'takeGoodPicturePoint',
                    'betweenProductInteractPoint', 'diversityPoint'])
        #投票により得られた得点
        y_label = [10, 6, 8, 5, 6, 7, 6, 12, 9, 9, 8, 4, 2, 5, 3, 0, 2, 0, 1, 0]
        y_dict = {}

        return y_data, y_label

class DataAugmentation:
    def makeNewFeature(self, data):
        communicationPoint = data['betweenProductInteractPoint'] + data['diversityPoint']
        inPhotoPoint = data['rareEncountPoint'] + data['takenPictureWithManyPeaplePoint']
        laughPicturePoint = data['laughStd'] + data['takenPictureWithManyPeaplePoint']

        data['communicationPoint'] = communicationPoint
        data['inPhotoPoint'] = inPhotoPoint
        data['laughPicturePoint'] = laughPicturePoint

        return data

    def execStd(self, data):
        stdData = (data - data.mean()) / data.std(ddof=0)
        print(stdData)

        return stdData

if __name__ == '__main__':

    data, label = makeTrainData()
    d = DataAugmentation()

    data = d.makeNewFeature(data)
    data = d.execStd(data)


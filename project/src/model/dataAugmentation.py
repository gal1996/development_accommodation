import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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

        return stdData

    def execPca(self, data):
        data = data.T
        pca = PCA(n_components=3)
        pca.fit_transform(data)
        pcaData = pca.components_
        pcaDf = pd.DataFrame(data = pcaData)
        pcaDf = pcaDf.T

        return pcaDf


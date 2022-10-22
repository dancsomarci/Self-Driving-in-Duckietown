from ast import Constant
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_processor import DataProcessor
import os
import constants

import cv2

def script():
    dp = DataProcessor()
    data = dp.get_list_of_frames_from_file(os.path.join(constants.processed_data_path, "processedData"))

    trainIdx = int(constants.train_data_size*len(data))
    valIdx = trainIdx + int(constants.validation_data_size*len(data))
    trainData = data[:trainIdx]
    valData = data[trainIdx: valIdx]
    testData = data[valIdx:]

    trainY, trainX = [label for label,_ in trainData], [img.ravel() for _,img in trainData]
    valY, valX = [label for label,_ in valData], [img.ravel() for _,img in valData]
    testY, testX = [label for label,_ in testData], [img.ravel() for _,img in testData]

    scaler = StandardScaler()
    scaler.fit(trainX)
    scaler.transform(trainX)
    scaler.transform(valX)
    scaler.transform(testX)

if __name__ == "__main__":
    script()
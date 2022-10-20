from matplotlib.pyplot import gray
from data_processor import DataProcessor, DataSplitter
import cv2
import numpy as np
import os
import constants

dp = DataProcessor()
dsplitter = DataSplitter()

# make a directory for processed data
if not os.path.isdir(constants.processed_data_path):
    os.mkdir(constants.processed_data_path)

# make a directory for splitted data
if not os.path.isdir(constants.split_data_path):
    os.mkdir(constants.split_data_path)


def handler(frame):
    '''preprocesses an image'''

    label, img = frame

    #cv2.imshow("Original", img)
    #cv2.waitKey(5)
    
    scale_precent = 0.2     # used for downscaling
    crop_amount = 35        # pixel amount for cropping

    # downscaling
    newHeight = int(img.shape[0] * scale_precent)
    newWidth = int(img.shape[1] * scale_precent)
    down_scaled_img = cv2.resize(img, (newHeight, newWidth))

    # cropping
    ds_size = down_scaled_img.shape
    cropped_img = down_scaled_img[crop_amount : ds_size[0], 0 : ds_size[1]]

    # grayscaling
    grayscale_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # thresholding
    threshold, thresholded_img = cv2.threshold(grayscale_img, 150, 255, cv2.THRESH_BINARY)

    # normalization
    mean = np.mean(thresholded_img)
    std = np.std(thresholded_img)
    normalized_img = (thresholded_img - mean) / std
    
    # showing image
    #print(normalized_img.shape)
    #cv2.imshow(str(label), normalized_img)
    #cv2.waitKey()

    processedFrame = (label, normalized_img)

    # as we have a new processed frame, we can just go ahead and put it into a dataset
    dsplitter.store_frame(processedFrame)

    return processedFrame

# processing every raw data file
dp.for_each_frame_from_file(constants.raw_data_path, constants.processed_data_path, handler)

# saving the processed and splitted files
dsplitter.persist_memory(constants.split_data_path)
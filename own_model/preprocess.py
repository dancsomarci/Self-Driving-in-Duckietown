from data_processor import DataProcessor
import cv2
import numpy as np
import os
import constants

def imageTransform(img):
    img = cv2.resize(img, (480, 640)) # make it the same size as the default size for observations in the duckietown env

    scale_precent = 0.2     # used for downscaling
    crop_amount = 35        # pixel amount for cropping

    # downscaling
    newHeight = int(img.shape[0] * scale_precent)
    newWidth = int(img.shape[1] * scale_precent)
    down_scaled_img = cv2.resize(img, (newHeight, newWidth))

    # cropping
    ds_size = down_scaled_img.shape
    cropped_img = down_scaled_img[crop_amount:,:]

     # grayscaling
    grayscale_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)

    # thresholding
    _, thresholded_img = cv2.threshold(grayscale_img, 150, 255, cv2.THRESH_BINARY)

    # binary image
    normalized_img = thresholded_img / 255

    return normalized_img

def script():
    dp = DataProcessor()

    # make a directory for processed data
    if not os.path.isdir(constants.processed_data_path):
        os.mkdir(constants.processed_data_path)

    def handler(frame):
        '''preprocesses an image'''

        label, img = frame
        img = imageTransform(img)
        processedFrame = (label, np.array(img))

        # as we have a new processed frame, we can just go ahead and put it into a dataset
        dp.store_frame(processedFrame)

    for frame in dp.frames(constants.raw_data_path):
        handler(frame) 

    dp.shuffle_frames()
    dp.persist_memory(constants.processed_data_path, "processedData")

if __name__ == "__main__":
    script()

from data_processor import DataProcessor
import cv2
import numpy as np
import os
import constants

def script():
    dp = DataProcessor()

    # make a directory for processed data
    if not os.path.isdir(constants.processed_data_path):
        os.mkdir(constants.processed_data_path)

    # make a directory for splitted data
    if not os.path.isdir(constants.split_data_path):
        os.mkdir(constants.split_data_path)

    def handler(frame):
        '''preprocesses an image'''

        label, img = frame
        
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
        grayscale_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)

        # thresholding
        _, thresholded_img = cv2.threshold(grayscale_img, 150, 255, cv2.THRESH_BINARY)

        processedFrame = (label, np.array(thresholded_img))

        # as we have a new processed frame, we can just go ahead and put it into a dataset
        dp.store_frame(processedFrame)

    for frame in dp.frames(constants.raw_data_path):
        handler(frame) 

    dp.shuffle_frames()
    dp.persist_memory(constants.processed_data_path, "processedData")

if __name__ == "__main__":
    script()

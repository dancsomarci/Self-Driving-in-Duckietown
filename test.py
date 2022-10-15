from matplotlib.pyplot import gray
from data_processor import DataProcessor
import cv2
import numpy as np

dp = DataProcessor()


def handler(frame):
    '''preprocesses an image'''
    label, img = frame

    cv2.imshow("Original", img)
    cv2.waitKey(5)
    
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
    print(normalized_img.shape)
    cv2.imshow(str(label), normalized_img)
    cv2.waitKey()

dp.for_each_frame_from_file("savedData", handler)
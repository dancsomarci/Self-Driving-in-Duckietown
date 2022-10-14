from data_processor import DataProcessor
import cv2

dp = DataProcessor()

def handler(frame):
    label, img = frame
    
    # TODO implement image processing
    cv2.imshow(str(label), img)
    cv2.waitKey()

dp.for_each_frame_from_file("savedData", handler)

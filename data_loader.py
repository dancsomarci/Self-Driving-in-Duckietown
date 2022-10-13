import os
import numpy as np
from PIL import Image
import random

path_to_folder = "savedData"
for filename in os.listdir(path_to_folder):
    file = open(os.path.join(path_to_folder, filename), "rb")
    byte_stream = file.read()
    for byte_frame in byte_stream.split(b'\n'):
        frame = np.frombuffer(byte_frame)
        print(frame.shape)
        im = Image.fromarray(frame)
        im.save("test//{}.jpeg".format(random.random()))
        break
    break



    ## TODO ##
    # image processing (opencv python:)
    # change format to be ready for learning
    


    
       

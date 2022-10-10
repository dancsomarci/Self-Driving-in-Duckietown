import os
import numpy as np

for filename in os.listdir("screenshots"):
    array = np.load(os.path.join("screenshots", filename)) # image as ndarray
    action = filename.split("#")[0:2] # action taken by driver

    ## TODO ##
    # image processing (opencv python:)
    # change format to be ready for learning
    


    
       

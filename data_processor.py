from array import array
from datetime import datetime
import os
import pickle
import random
import constants
import numpy as np

class DataProcessor:
    def __init__(self):
        self.memory = []
        random.seed(constants.random_numpy_seed)

    def store_frame(self, frame):
        self.memory.append(frame)

    def __flush_memory(self):
        self.memory = []

    def persist_memory(self, save_path: str, filename=datetime.now().timestamp()):
        file = open(os.path.join(save_path, filename), "wb")

        pickle.dump(self.memory, file)
        file.close()
        self.__flush_memory() #!!!

    def frames(self, data_path):
        # process raw data with generator
        for filename in os.listdir(data_path):
            file = open(os.path.join(data_path, filename), "rb")
            for frame in pickle.load(file):
                yield frame

    def for_each_frame_from_file(self, data_path, func_for_each_frame):
        # process raw data with handler function
        for frame in self.frames(data_path):
            processedFrame = func_for_each_frame(frame)
            self.store_frame(processedFrame)

    def shuffle_frames(self):
        np.random.shuffle(self.memory)

    def get_list_of_frames_from_file(self, file_path) -> array:
        file = open(file_path, "rb")
        return pickle.load(file)

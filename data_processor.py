from datetime import datetime
import os
import pickle

class DataProcessor:
    def __init__(self):
        self.memory = []
        
    def store_frame(self, frame):
        self.memory.append(frame)

    def flush_memory(self):
        self.memory = []

    def persist_memory(self, save_path: str):
        file = open(os.path.join(save_path, "{}".format(datetime.now().timestamp())), "wb")
        pickle.dump(self.memory, file)
        file.close()
        self.flush_memory() #!!!

    def for_each_frame_from_file(self, save_path, func_for_each_frame):
        for filename in os.listdir(save_path):
            file = open(os.path.join(save_path, filename), "rb")
            for frame in pickle.load(file):
                func_for_each_frame(frame)
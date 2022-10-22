from datetime import datetime
import os
import pickle
import random
import constants

class DataProcessor:
    def __init__(self):
        self.memory = []
        
    def store_frame(self, frame):
        self.memory.append(frame)

    def __flush_memory(self):
        self.memory = []

    def persist_memory(self, save_path: str):
        file = open(os.path.join(save_path, "{}".format(datetime.now().timestamp())), "wb")

        pickle.dump(self.memory, file)
        file.close()
        self.__flush_memory() #!!!

    def frames(self, data_path):
        for filename in os.listdir(data_path):
            file = open(os.path.join(data_path, filename), "rb")
            for frame in pickle.load(file):
                yield frame

    def for_each_frame_from_file(self, data_path, save_path, func_for_each_frame):
        # process every raw data, and then save the processed data
        for frame in self.frames(data_path):
            processedFrame = func_for_each_frame(frame)
            self.store_frame(processedFrame)

        self.persist_memory(save_path) #ez miért kell, ha van dataSplitter is? szerintem elég lenne itt kimenteni majd tanításnál bekeverni

    
class DataSplitter():
    '''Splits the data into 3 categories, and saves them to their respective file.'''

    def __init__(self):
        self.train_memory = []
        self.test_memory = []
        self.validation_memory = []
        random.seed(constants.random_seed_for_split)

    def store_frame(self, frame):
        '''Sorting the frame into one of the categories, and saving it to memory.'''
        chance = random.random()        # generates a random number [0,1)
        if chance < constants.train_data_size:
            self.train_memory.append(frame)
        elif chance < constants.train_data_size + constants.test_data_size:
            self.test_memory.append(frame)
        else:
            self.validation_memory.append(frame)
        
    def __flush_memory(self):
        self.train_memory.clear()
        self.test_memory.clear()
        self.validation_memory.clear()

    def __save_to_file(self, save_path, file_name, memory):
        file = open(os.path.join(save_path, file_name), "wb")
        pickle.dump(memory, file)
        file.close()

    def persist_memory(self, save_path: str):
        '''Saving memmories to files.'''
        self.__save_to_file(constants.split_data_path, constants.train_data_name, self.train_memory)
        self.__save_to_file(constants.split_data_path, constants.test_data_name, self.test_memory)
        self.__save_to_file(constants.split_data_path, constants.validation_data_name, self.validation_memory)
        self.__flush_memory()
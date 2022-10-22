# the paths to the different data files (they should be directory names only!)
raw_data_path = 'savedData'
processed_data_path = 'processedData'
split_data_path = 'splitData'

# constant seed, so the result is reproducable
random_numpy_seed = 1234

# these are chances, together they should add up to 1
train_data_size = 0.8
test_data_size = 0.1
validation_data_size = 0.1

# the file names used for the 3 different data type
train_data_name = 'training_data'
test_data_name = 'testing_data'
validation_data_name = 'validation_data'
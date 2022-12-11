# Self-Driving-in-Duckietown

The project is created as an assignment for the VITMAV45 University course at BME.

# Team: TescosTesla

Members:

- Marcell Dancsó - AZTVS7
- Milán Nyist - VU9J1J
- Dániel Veress - C8P32R

# About the project:

Making a bot learn in a simulated environment to follow the lane as accurately as possible with different deep learning techniques.

# Links:

- Here is a preview of the controller, while driving on our own map: https://youtu.be/yeaDMzU-XvI
- Drive link, where we store the processed data: https://drive.google.com/drive/folders/1qC-w2b-WBtoXA9E3Cql4zntL-kaU6D7D?usp=sharing
  - Data will be added later on to reach a big enough dataset

# Project structure:

- gym_duckietown : modified gym environment for training
- driving : helpers for handling user input during driving
- own_model : code for imitationlearning
- baselines : industry standard algorithms for solving the problem


# Setup

Create a conda environment from `environment.yml`:
````
conda env create -f environment.yml
````
After you can run the desired files in the subdirectories.

# Driving

For testing out maps, and the simulation environment with an easy to use, and ergonomic controls navigate to the `driving` folder:
````
python manual.py [--map-name "map_name"]
````
Running the script will present you with a graphical view to the simulator. Move the agent with the "wasd" keys, press `esc` to quit. An optional parameter is the map name which must have a corresponding `maps\map_name.yml` in the current working directory where the script was launched.

# Imitation learning

## Files

The related files can be found in the `own_model` folder.

- constans.py : constants for preprocessing the data
- data_processor.py : handles saving of frames from simulator
- datarecorder.py : simulator variant where humans can generate data for imitation learning purposes
- imitation_learning_training.ipynb : training of the imitation learning models (contains experiments with different models and architectures)
- preprocess.py : image processing with opencv
- test_model.py : testing the trained model
- visualization.ipynb : visualization of the image processing steps

## Testing

For testing our model, choose one from the imitation_learning_training.ipynb file and generate the `.hdf5` file. Another option is to download our already trained model from:
```
https://drive.google.com/drive/folders/1NMN81yTgm0qJNJlXYJEBY0FYbkZKUhZa?usp=share_link
```
After downloading the `.hdf5` set the path in the `test_model.py` and run the script.
You can controll the agent manually with "wasd" to setup the desired starting position and press `p` to let the agent take over.

For changing maps check out the [driving](#driving) section above. The `test_model.py` script can be parametrized just as `manual.py`.

# Baseline solutions

The related files can be found in the `baselines` folder.
There are 2 files for each baseline algorithm:

- baseline_training.py : handles the training process
- baseline_test.py : testing the taught model

There's also an additional file called `wrappers.py` which contains the necessary extensions for the basic environment for modifying actions, rewards and observations.

For testing the models traing your own parameters via the `baseline_training.py` or download our solutions from the following link:
````
https://drive.google.com/drive/folders/1NMN81yTgm0qJNJlXYJEBY0FYbkZKUhZa?usp=share_link
````

# PPO
````
https://www.youtube.com/watch?v=eLKxaiax6Ks&ab_channel=NyistMilan
````

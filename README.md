# Self-Driving-in-Duckietown

The project is created as an assignment for the VITMAV45 University course at BME.

# Team: TescosTesla

Members:

- Marcell Dancsó - AZTVS7
- Milán Nyist - VU9J1J
- Dániel Veress - C8P32R

# About the project:

Making a bot learn in a simulated environment to follow the lane as accurately as possible with different deep learning techniques.

# Test Driving:

Here is a preview of the controller, while driving on our own map: https://youtu.be/yeaDMzU-XvI 

# Files:

- my_control.py
  - starts a simulated environment
- conroller.py
  - responsible for controlling the environment
  - acquire data from it
- data_processor.py
  - saves the data to memory
  - splits the data into training, testing and validation parts
- preprocess.py
  - preprocesses raw data with opencv library
- Visualisation.ipynb
  - notebook to show the dataset visualised
-splitdata_demo.py
  - demo for using the preprocessed data, and normalizing it for training
- maps directory
  - the different maps that the bot can run on
- gym_duckietown directory
  - the different file for the simulation environment
    (the files are taken from the following publicly available repo: https://github.com/duckietown/gym-duckietown/tree/daffy/src/gym_duckietown)

# How to run:

1. Start a console.
2. Navigate to the directory where my_control.py is located.
3. To run the simulation: ./my_control.py [--map-name map_name]
   - To start or stop collection data: press P or left Shift once
   - To exit: press escape
4. To start preprocessing and splitting of raw data: ./preprocess.py
   - This will downscale, crop etc. the image
   - This will also split the data into training, testing and validation parts

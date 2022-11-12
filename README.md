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
  - Data will be added later on to reach a big enough dataset.

# Files:

- my_control.py
  - starts a simulated environment
- conroller.py
  - responsible for controlling the environment
  - acquire data from the environment
- data_processor.py
  - saves the data to memory
- preprocess.py
  - preprocesses raw data with opencv library
- Visualisation.ipynb
  - notebook to show the dataset visualised
- A2C_Baseline.ipynb
  - notebook for A2C reinforcement learning algorithm
- PPO_Baseline.ipynb
  - notebook for PPO reinforcement learning algorithm
- splitdata_demo.py
  - demo for using the preprocessed data, and normalizing it for training
- constants.py
  - a file for constant values (e.g, data paths)
- env.py
  - create the environment where the reinforcement learning agent will be learning
- wrappers.py
  - environment wrappers to modify the base environment
- maps directory
  - the different maps that the bot can run on
  - our_road.yaml - the road that we made for training
- gym_duckietown directory
  - the different file for the simulation environment
    (the files are taken from the following publicly available repo: https://github.com/duckietown/gym-duckietown/tree/daffy/src/gym_duckietown)

# How to run:

1. Start a console.
2. Navigate to the directory where my_control.py is located.
3. To run the simulation: ./my_control.py [--map-name map_name]
   - To start or stop collecting data: press P or left Shift once
   - To exit: press escape
4. To start the preprocessing of raw data: ./preprocess.py
   - This will downscale, crop etc. the images, shuffle them, and store them presistantly to the folder defined in constants.py as "processed_data_path".
5. To split the data into training, testing and validation sets: ./splitdata_demo.py
   - This will split the data and demonstrate the normalization process.

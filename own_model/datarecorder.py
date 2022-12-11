# Global import
import os
from pyglet.window import key

# Own modules
from data_processor import DataProcessor
import constants

# Own modules from parallel directory
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from driving.driving_controller import ManualDrivingController
from driving.duckietown_env_argparser import parser_args

class DataRecorderController(ManualDrivingController):
    '''Handles the gym environment and controls for the bot, while giving the driver the oprtion to gather data for learning. (press p to record)'''

    def __init__(self, args):
        super().__init__(args)

        self.camera = DataProcessor()
        self.is_recording = False
        self.save_path = constants.raw_data_path

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        self.env.render() # this line must be before registering hadles

        @self.env.unwrapped.window.event
        def on_key_press(symbol, _):
            if symbol == key.BACKSPACE:
                self.resetEnv()

            if symbol == key.ESCAPE:
                self.stop()
            
            if symbol == key.P or symbol == key.LSHIFT:
                if self.is_recording:
                    if input("Do you want to save the data? (y/n) ") == "y":
                        self.camera.persist_memory(self.save_path)
                else:
                    print("Recording in progress...")
                self.is_recording = not self.is_recording
                   
        self.key_handler = key.KeyStateHandler() # these 2 lines must be after "def on_key_press" !!!
        self.env.unwrapped.window.push_handlers(self.key_handler)

    def update(self, dt):
        action = self.calc_input()

        if self.is_recording:
            self.camera.store_frame(((action[0], action[1]), self.env.render('rgb_array'))) #by default this gives a picture with width=800, height=600?

        obs, _, _, _ = self.env.step(action) #obs is width=640, height=480

        self.env.render()

def script():
    args = parser_args()
    controller = DataRecorderController(args)
    controller.start()

if __name__ == "__main__":
    script()
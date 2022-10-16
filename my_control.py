import argparse
import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import os

from data_processor import DataProcessor
import constants

# handling arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="small_loop")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--style", default="photos", choices=["photos", "synthetic", "synthetic-F", "smooth"])
args = parser.parse_args()


class Controller:
    '''Controls the bot, and saves data.'''

    def __init__(self):
        if args.env_name is None:
            self.env = DuckietownEnv(
                map_name=args.map_name,
                draw_curve=args.draw_curve,
                draw_bbox=args.draw_bbox,
                domain_rand=args.domain_rand,
                frame_skip=args.frame_skip,
                distortion=args.distortion,
                style=args.style,
            )
        else:
            self.env = gym.make(args.env_name)

        self.camera = DataProcessor()
        self.is_recording = False
        self.save_path = constants.raw_data_path

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        self.env.render()

        @self.env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.BACKSPACE: # BACKSPACE==RESET
                self.resetEnv()

            if symbol == key.ESCAPE:
                self.stop()
            
            if symbol == key.P or symbol == key.LSHIFT: # Start exporting when the P key or left-Shift is pressed
                if self.is_recording:
                    if input("Do you want to save the data? (y/n) ") == "y":
                        self.camera.persist_memory(self.save_path)
                else:
                    print("Recording in progress...")
                self.is_recording = not self.is_recording
                   
        self.key_handler = key.KeyStateHandler() # these 2 lines must be after "def on_key_press" !!!
        self.env.unwrapped.window.push_handlers(self.key_handler)

    def update(self, dt):
        action = np.array([0.0, 0.0])

        if self.key_handler[key.W]:
            action += np.array([1, 0])
        if self.key_handler[key.S]:
            action += np.array([-1, 0])
        if self.key_handler[key.A]:
            action += np.array([-0.5, 3])
        if self.key_handler[key.D]:
            action += np.array([-0.5, -3])
        if self.key_handler[key.SPACE]:
            action = np.array([0, 0])

        obs, reward, done, info = self.env.step(action) # obs is the image seen or maybe after taking the action but it shouldn't matter because of high framerate

        if self.is_recording:
            self.camera.store_frame(((action[0], action[1]), obs))

        self.env.render()

    def start(self):
        pyglet.clock.schedule_interval(self.update, 1.0 / self.env.unwrapped.frame_rate)
        pyglet.app.run()

    def stop(self):
        self.env.close()

    def resetEnv(self):
        self.env.reset()
        self.env.render()


controller = Controller()
controller.start()
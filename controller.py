import gym
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import os
import numpy as np

from data_processor import DataProcessor
import constants


class MyController:
    '''Handles the gym environment and controls for the bot.'''

    def __init__(self, args):
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

        # obs is the image seen after taking the action but it shouldn't matter because of high framerates
        obs, _, _, _ = self.env.step(action)

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
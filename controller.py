import gym
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
import os
import numpy as np
from tensorflow.keras.models import load_model
import cv2

from data_processor import DataProcessor
import constants

class Controller:
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

    def update(self, dt):
        pass

    def start(self):
        pyglet.clock.schedule_interval(self.update, 1.0 / self.env.unwrapped.frame_rate)
        pyglet.app.run()

    def stop(self):
        self.env.close()

    def resetEnv(self):
        self.env.reset()
        self.env.render()


class MyController(Controller):
    '''Handles the gym environment and controls for the bot.'''

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

class ModelController(Controller):
    '''Switch between manual driving and AI driving with "p"'''

    def __init__(self, args):
        super().__init__(args)

        self.model = load_model("firstNet.hdf5") # https://drive.google.com/file/d/1OXIHaQ3fCP97oAhGzKIZ0lAVdrob9_iz/view?usp=share_link
        self.model_control = False

        self.env.render() # this line must be before registering hadles

        @self.env.unwrapped.window.event
        def on_key_press(symbol, _):
            if symbol == key.BACKSPACE:
                self.resetEnv()

            if symbol == key.ESCAPE:
                self.stop()

            if symbol == key.P:
                self.model_control = not self.model_control

        self.key_handler = key.KeyStateHandler() # these 2 lines must be after "def on_key_press" !!!
        self.env.unwrapped.window.push_handlers(self.key_handler)

    def update(self, dt):
        action = np.array([0.0, 0.0])

        if self.model_control:
            img = self.env.render('rgb_array')
            scale_precent = 0.2
            crop_amount = 35
            newHeight = int(img.shape[0] * scale_precent)
            newWidth = int(img.shape[1] * scale_precent)
            down_scaled_img = cv2.resize(img, (newHeight, newWidth)) #TODO train on appropriate data
            ds_size = down_scaled_img.shape
            cropped_img = down_scaled_img[crop_amount : ds_size[0], 0 : ds_size[1]]
            grayscale_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
            _, thresholded_img = cv2.threshold(grayscale_img, 150, 255, cv2.THRESH_BINARY)
            final = cv2.resize(thresholded_img, (96, 93))

            action = self.model.predict(np.array([final]), verbose=0)[0]
        else:
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

        obs, reward, done, info = self.env.step(action)

        if done:
            self.resetEnv()

        self.env.render()
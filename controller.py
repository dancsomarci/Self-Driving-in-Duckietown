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
from preprocess import imageTransform

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

def calc_input(key_handler):
    action = np.array([0.0, 0.0])

    if key_handler[key.A] and key_handler[key.W]:
        action += np.array([0.5, 3])
    elif key_handler[key.D] and key_handler[key.W]:
        action += np.array([0.5, -3])
    elif key_handler[key.W]:
        action += np.array([1, 0])
    elif key_handler[key.S]:
        action += np.array([-1, 0])

    return action
    
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
        action = calc_input(self.key_handler)

        if self.is_recording:
            self.camera.store_frame(((action[0], action[1]), self.env.render('rgb_array'))) #by default this gives a picture with width=800, height=600?

        obs, _, _, _ = self.env.step(action) #obs is width=640, height=480

        self.env.render()

class ModelController(Controller):
    '''Switch between manual driving and AI driving with "p"'''

    def __init__(self, args):
        super().__init__(args)

        self.model = load_model(constants.classification_model_weights_filename) # https://drive.google.com/file/d/1x99W6f25oaPZ31KXvpi2FhTWilBfNJe7/view?usp=share_link for the good weights or run the own_model_training.ipynb to acquire the file

        # for the classification model
        self.inverse_transform_to_categorical = {
            (0.,1.,0.): np.array([1.,0.]),
            (1.,0.,0.): np.array([.5,3.]),
            (0.,0.,1.): np.array([.5,-3.]),
            (0.,0.,0.): np.array([0.,0.])
        }

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
            final = imageTransform(img)
            action = self.model.predict(np.array([final]), verbose=0)[0]

            # if classification
            temp = [0.,0.,0.]
            idx = np.argmax(action)
            temp[idx] = 1.
            action = self.inverse_transform_to_categorical[tuple(temp)]
        else:
            action = calc_input(self.key_handler)

        obs, reward, done, info = self.env.step(action)

        if done:
            self.resetEnv()

        self.env.render()
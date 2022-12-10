# Global import
import os
from pyglet.window import key
from tensorflow.keras.models import load_model
import numpy as np

# Own modules
import constants
from preprocess import imageTransform

# Own modules from parallel directory
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from driving.driving_controller import ManualDrivingController
from driving.duckietown_env_argparser import parser_args

class ImitationLearningController(ManualDrivingController):
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
            action = self.calc_input()

        obs, reward, done, info = self.env.step(action)

        if done:
            self.resetEnv()

        self.env.render()


def script():
    args = parser_args()
    controller = ImitationLearningController(args)
    controller.start()

if __name__ == "__main__":
    script()
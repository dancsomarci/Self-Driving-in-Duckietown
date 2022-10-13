
import argparse
import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--style", default="photos", choices=["photos", "synthetic", "synthetic-F", "smooth"])
args = parser.parse_args()

class Camera:
    def __init__(self):
        self.file = None

    def is_ready(self):
        return self.file is not None

    def init(self, save_path):
        self.file = open(os.path.join(save_path, "{}".format(datetime.now().timestamp())), "ab")

    # frame is an ndarray
    def save_frame(self, frame):
        if self.file is None:
            return
        self.file.write(frame.tobytes())
        self.file.write("\n".encode('utf-8'))

    def save_film(self):
        if self.is_ready():
            self.file.close()
            self.file = None

    def delete_film(self):
        if self.is_ready():
            self.file.close()
            os.remove(self.file.name)
            self.file = None


class Controller:
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

        self.camera = Camera()
        self.save_path = "savedData" #TODO set!!

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
                if not self.camera.is_ready():
                    self.camera.init(self.save_path)
                else:
                    if input("Do you want to save the data? (y/n)") == "y":
                        self.camera.save_film()
                    else:
                        self.camera.delete_film()

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

        self.camera.save_frame(obs)
        print(obs.shape)

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

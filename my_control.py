
import argparse
import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from datetime import datetime
import os

savepath = "foldername" #TODO set!!

if not os.path.isdir(savepath):
    os.mkdir(savepath)

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

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        style=args.style,
    )
else:
    env = gym.make(args.env_name)

def resetEnv():
    env.reset()
    env.render()

resetEnv()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    if symbol == key.BACKSPACE: # BACKSPACE==RESET
        resetEnv()

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

import csv

class Timer():
    def __init__(self) -> None:
        self.time = 0.0

def update(dt):
    action = np.array([0.0, 0.0])
    timer.time += dt

    if key_handler[key.UP] or key_handler[key.W]:
        action += np.array([1, 0])
    if key_handler[key.DOWN] or key_handler[key.S]:
        action += np.array([-1, 0])
    if key_handler[key.LEFT] or key_handler[key.A]:
        action += np.array([-0.5, 3])
    if key_handler[key.RIGHT] or key_handler[key.D]:
        action += np.array([-0.5, -3])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    obs, reward, done, info = env.step(action) # obs is the image seen or maybe after taking the action but it shouldn't matter because of high framerate

    if timer.time >= 5.0: # start exporting after 5 seconds
        np.save(os.path.join(savepath, "{}#{}#{}".format(action[0], action[1], datetime.now().timestamp())), obs)

    env.render()

timer = Timer()
pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
pyglet.app.run()
env.close()

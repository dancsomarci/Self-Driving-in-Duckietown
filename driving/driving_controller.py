import pyglet
from pyglet.window import key
import numpy as np


import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from gym_duckietown.envs import DuckietownEnv

class DrivingController:
    def __init__(self, args):
        self.env = DuckietownEnv(
            map_name=args.map_name,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            frame_skip=args.frame_skip,
            distortion=args.distortion,
            style=args.style,
        )

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


class ManualDrivingController(DrivingController):
    '''Handles the gym environment and controls for the bot.'''

    def __init__(self, args):
        super().__init__(args)

        self.env.render() # this line must be before registering hadles

        @self.env.unwrapped.window.event
        def on_key_press(symbol, _):
            if symbol == key.BACKSPACE:
                self.resetEnv()

            if symbol == key.ESCAPE:
                self.stop()

        self.key_handler = key.KeyStateHandler() # these 2 lines must be after "def on_key_press" !!!
        self.env.unwrapped.window.push_handlers(self.key_handler)

    def calc_input(self):
        action = np.array([0.0, 0.0])

        if self.key_handler[key.A] and self.key_handler[key.W]:
            action += np.array([0.5, 3])
        elif self.key_handler[key.D] and self.key_handler[key.W]:
            action += np.array([0.5, -3])
        elif self.key_handler[key.W]:
            action += np.array([1, 0])
        elif self.key_handler[key.S]:
            action += np.array([-1, 0])

        return action

    def update(self, dt):
        action = self.calc_input()

        obs, _, _, _ = self.env.step(action) #obs is width=640, height=480

        self.env.render()
    
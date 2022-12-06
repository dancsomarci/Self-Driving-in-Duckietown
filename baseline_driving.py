import argparse
from stable_baselines3 import A2C

from env import a2c_env
from driving_controller import BaseLineModelController

def script():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--map-name", default="our_road"); # our road is the default road
    parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
    parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
    parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
    parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
    parser.add_argument("--distortion", default=False, action="store_true")
    parser.add_argument("--style", default="photos", choices=["photos", "synthetic", "synthetic-F", "smooth"])
    args = parser.parse_args()

    model = A2C.load("..\\Self-Driving-in-Duckietown\\models\\a2c_model_40000_steps")
    
    controller = BaseLineModelController(args, model, a2c_env())
    controller.start()

if __name__ == "__main__":
    script()
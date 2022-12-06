import gym
from gym_duckietown.simulator import Simulator

from wrappers import (
    NormalizeWrapper,
    ResizeWrapper,
    ClipImageWrapper,
    RGB2GrayscaleWrapper,
    DtRewardWrapperDistanceTravelled,
    DtRewardCollisionAvoidance,
    A2CActionWrapper,
)

def launch_env(id=None, map="loop_empty"):
    env = None
    if id is None:
        env = Simulator(
            seed=123,
            map_name=map,
            max_steps=500,
            domain_rand=True, #Set to False if testing...
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,
            full_transparency=True,
            frame_rate=30,
            distortion=True,
        )
    else:
        env = gym.make(id)

    # Wrappers
    env = ClipImageWrapper(env)
    env = RGB2GrayscaleWrapper(env)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = DtRewardWrapperDistanceTravelled(env)
    env = DtRewardCollisionAvoidance(env)
    return env

def a2c_env(map="loop_empty"):
    env = Simulator(
                seed=123,
                map_name=map,
                max_steps=500,
                domain_rand=True, #Set to False if testing...
                camera_width=640,
                camera_height=480,
                accept_start_angle_deg=4,
                full_transparency=True,
                frame_rate=30,
                distortion=True,
            )
    env = ClipImageWrapper(env)
    env = RGB2GrayscaleWrapper(env)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = DtRewardWrapperDistanceTravelled(env)
    env = DtRewardCollisionAvoidance(env)
    env = A2CActionWrapper(env)
    return env
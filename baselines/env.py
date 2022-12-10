import gym
from gym_duckietown.simulator import Simulator

from wrappers import (
    NormalizeWrapper,
    ResizeWrapper,
    ClipImageWrapper,
    RGB2GrayscaleWrapper,
    DtRewardWrapperDistanceTravelled,
    A2CActionWrapper,
    DQNActionWrapper,
    MyObservationWrapper,
    DtRewardPosAngle,
    DtRewardTargetOrientation,
    DtRewardClipperWrapper,
    DtRewardVelocity,
    InconvenientSpawnFixingWrapper,
    DQNActionWrapperSimple
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
    env = A2CActionWrapper(env)
    return env

from gym import spaces

class DiscreteDuckieTown(Simulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.action_space = spaces.Discrete(3)

def dqn_env(map="loop_empty"):
    env = DiscreteDuckieTown(
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
    env = MyObservationWrapper(env)
    env = DQNActionWrapper(env)
    return env

def master_dqn_env(map="our_road"):
    env = DiscreteDuckieTown(
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
    env = DQNActionWrapperSimple(env)
    env = MyObservationWrapper(env)
    env = DtRewardWrapperDistanceTravelled(env)
    env = DtRewardPosAngle(env)
    env = DtRewardTargetOrientation(env)
    env = DtRewardClipperWrapper(env)
    env = DtRewardVelocity(env)
    env = InconvenientSpawnFixingWrapper(env)
    return env

def master_dqn_env_testing(map="our_road"):
    env = DiscreteDuckieTown(
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
    env = DQNActionWrapperSimple(env)
    return env

def master_env(map="our_road"):
    env = Simulator(
                seed=123,
                map_name=map,
                max_steps=500,
                domain_rand=False, #Set to False if testing...
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
    env = DtRewardPosAngle(env)
    env = DtRewardTargetOrientation(env)
    env = DtRewardClipperWrapper(env)
    env = DtRewardVelocity(env)
    env = InconvenientSpawnFixingWrapper(env)
    return env
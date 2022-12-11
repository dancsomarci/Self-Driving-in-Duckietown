from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers import (
    NormalizeWrapper,
    ResizeWrapper,
    ClipImageWrapper,
    RGB2GrayscaleWrapper,
    DtRewardWrapperDistanceTravelled,
    A2CActionWrapper,
)

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from gym_duckietown.simulator import Simulator

def a2c_env():
  env = Simulator(
                seed=123,
                map_name="our_road_extreme",
                max_steps=500,
                domain_rand=True,
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

env = DummyVecEnv([a2c_env])

a2c_model = A2C(policy='MlpPolicy',
                env=env,
                learning_rate=0.0003,
                n_steps=2048,
                use_rms_prop=False,
                gamma=0.99,
                gae_lambda=0.95,
                normalize_advantage=True,
                ent_coef=0.0,
                vf_coef=0.5,
                verbose=1,
                device="auto",
                seed=123)

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="a2c_model_saves\\",
  name_prefix="a2c_model",)

a2c_model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)
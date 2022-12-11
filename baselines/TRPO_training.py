from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from gym_duckietown.simulator import Simulator

from wrappers import (
    NormalizeWrapper,
    ResizeWrapper,
    ClipImageWrapper,
    RGB2GrayscaleWrapper,
    DtRewardWrapperDistanceTravelled,
)

def trpo_env():
  env = Simulator(
            seed=123,
            map_name="our_road_extreme",
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
  return env

env = DummyVecEnv([trpo_env])

trpo_model = TRPO(policy='MlpPolicy',
                env=env,
                learning_rate=0.001,
                n_steps=2048,
                batch_size=128,
                gamma=0.99,
                cg_max_steps=15,
                cg_damping=0.1,
                line_search_shrinking_factor=0.8,
                line_search_max_iter=10,
                n_critic_updates=10,
                gae_lambda=0.95,
                use_sde=False,
                sde_sample_freq=-1,
                normalize_advantage=True,
                target_kl=0.01,
                sub_sampling_factor=1,
                tensorboard_log=None,
                policy_kwargs=None,
                verbose=1,
                seed=123,
                device='auto',
                _init_setup_model=True
            )

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="trpo_model_saves\\",
  name_prefix="trpo_model",)

trpo_model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)
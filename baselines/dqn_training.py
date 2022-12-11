from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
from wrappers import (
    DQNActionWrapperSimple,
    PreprocessForCnnObservationWrapper
)

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from gym_duckietown.simulator import Simulator

class DiscreteDuckieTown(Simulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.action_space = spaces.Discrete(3)

def dqn_env():
  env = DiscreteDuckieTown(
                seed=123,
                map_name="our_road_extreme",
                max_steps=500,
                domain_rand=False,
                camera_width=640,
                camera_height=480,
                accept_start_angle_deg=4,
                full_transparency=True,
                frame_rate=30,
                distortion=True,
            )
  env = PreprocessForCnnObservationWrapper(env)
  env = DQNActionWrapperSimple(env)
  return env

def script():
  env = DummyVecEnv([dqn_env])

  dqn_model = DQN(policy='CnnPolicy',
                  env=env,
                  buffer_size=20000,
                  learning_starts=20000,
                  tensorboard_log="tb_logs\\dqn\\",
                  seed=123)

  checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="dqn_model_saves\\",
    name_prefix="dqn_model",)

  dqn_model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)

if __name__ == "__main__":
    script()
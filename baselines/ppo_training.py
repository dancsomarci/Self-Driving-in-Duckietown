from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from wrappers import (
    NormalizeWrapper,
    ResizeWrapper,
    ClipImageWrapper,
    RGB2GrayscaleWrapper,
    DtRewardWrapperDistanceTravelled,
    DtRewardCollisionAvoidance,
)

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from gym_duckietown.simulator import Simulator


def ppo_env():
  env = Simulator(
    seed=123,
    map_name="loop_empty",
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

  return env


def script():
  env = ppo_env()

  ppo_model = PPO(policy='MlpPolicy',
                  env=env,
                  learning_rate=5.e-5,
                  gamma=0.99,
                  tensorboard_log="tb_logs\\ppo\\",
                  verbose=1)

  checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="ppo_model_saves\\",
    name_prefix="ppo_model",)

  ppo_model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)

if __name__ == "__main__":
    script()




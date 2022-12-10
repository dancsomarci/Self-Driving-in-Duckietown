from env import master_dqn_env

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([master_dqn_env])

dqn_model = DQN(policy='MlpPolicy',
                env=env,
                buffer_size=20000,
                learning_starts=20000,
                seed=123)

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="..\\Self-Driving-in-Duckietown\\models\\",
  name_prefix="dqn_model",)

dqn_model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)
from env import a2c_env

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

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
  save_freq=10000,
  save_path="..\\Self-Driving-in-Duckietown\\models\\",
  name_prefix="a2c_model",)

a2c_model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)
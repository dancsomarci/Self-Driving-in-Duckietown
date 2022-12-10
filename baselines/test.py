from env import master_dqn_env

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([master_dqn_env])
model = DQN.load("..\\Self-Driving-in-Duckietown\\models\\dqn_model_230000_steps")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

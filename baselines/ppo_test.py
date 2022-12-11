from ppo_training import ppo_env

from stable_baselines3 import PPO

env = ppo_env()
model = PPO.load("ppo_model_saves\\ppo_model_1_steps")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

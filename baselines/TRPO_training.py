from env import master_env

from sb3_contrib import TRPO
#from stable_baselines3 import TRPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([master_env])

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
  save_freq=10,
  save_path="..\\Self-Driving-in-Duckietown\\models\\",
  name_prefix="trpo_model",)

trpo_model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)
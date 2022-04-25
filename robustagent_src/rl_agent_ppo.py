from pathlib import Path
from typing import List
from stable_baselines3 import PPO
import torch as th
import hyperparam as hp
import rl_agent_base as base


def train(lr_start: float, gamma: float, training_steps: int, steps_per_update: int, model_no=-1):
    env_norm = base.get_env(n_steps=training_steps, learning_rate_start=lr_start)
    model = PPO(
        "MlpPolicy",  # actor critic
        env_norm,
        tensorboard_log=hp.TENSORBOARD_LOG_PATH,
        policy_kwargs=base.NET_ARCH,
        verbose=1,
        device="cuda",
        learning_rate=base.linear_schedule(lr_start),
        gamma=gamma,
        n_steps=steps_per_update,
        batch_size=steps_per_update,
    )
    base.train(model, "ppo", training_steps, model_no)


def test(test_episodes: int, result_suffix: str, sample_ids: List[int], model_no=-1):
    env_norm = base.get_env(n_steps=test_episodes, sample_ids=sample_ids)
    model_info = base.get_model_name("ppo", model_no, with_model_path=True)
    base.test(env_norm, PPO.load(model_info['model_path']), model_info['model_name'], result_suffix, test_episodes)

from pathlib import Path
from typing import List
from stable_baselines3 import A2C
import torch as th
import hyperparam as hp
import rl_agent_base as base

def train(lr_start: float, gamma: float, training_steps: int, steps_per_update: int):
    env_norm = base.get_env(training_steps, lr_start)
    model = A2C(
        "MlpPolicy",  # actor critic
        env_norm,
        tensorboard_log=hp.TENSORBOARD_LOG_PATH,
        policy_kwargs=base.NET_ARCH,
        verbose=1,
        device="cuda",
        learning_rate=base.linear_schedule(lr_start),
        gamma=gamma,
        n_steps=steps_per_update, 
        use_rms_prop=False
    )
    base.train(model, 'a2c', training_steps)


def test(test_episodes: int, result_suffix: str, sample_ids: List[int]):
    name = f"model_a2c_{hp.SCHED_OBJECTIVE}_J{4}"
    env_norm = base.get_env(n_steps=test_episodes, sample_ids=sample_ids)
    p = Path(".")
    p = p / "drl_models" / name
    base.test(env_norm, A2C.load(p), name, result_suffix, test_episodes)

# perform_tests = 1
# if perform_tests == 1:
#     test()
# else:
#     train()
